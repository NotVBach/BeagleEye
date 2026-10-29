#include "vision.h"
#include "camera.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csetjmp>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <pthread.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/syscall.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

extern "C" {
#include <jpeglib.h>
}

namespace fs = std::filesystem;

namespace {

const int    FRAME_SIDE    = 640;
const int    MAX_FACES     = 10;
const size_t MAX_NAME      = 64;
const float  NMS_THRESHOLD = 0.3f;
const int    STRIDES[]     = { 8, 16, 32 };
const char  *HEADS[]       = { "cls", "obj", "bbox", "kps" };

struct Known { std::string name; cv::Mat feature; };
struct Track { cv::Rect2f box; float score; std::string name; float sim; double recog_at; };
struct JpegErr { jpeg_error_mgr pub; jmp_buf jump; unsigned char *volatile buf; };

/* A TFLite model; the interpreter reads constants straight from file, so it must stay alive */
struct Net {
    std::vector<unsigned char>              file;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter>     it;
};

std::string detect_model, recog_model, faces_dir;
int   detect_w, detect_h;
float score_threshold, match_threshold, recog_interval;

Net  detector, recognizer;
bool have_recognizer;
int  heads[4][3];    /* YuNet output tensor index per head (cls, obj, bbox, kps) and stride */
std::vector<Known> known;

pthread_mutex_t lock    = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  updated = PTHREAD_COND_INITIALIZER;
std::string     result  = "{\"fps\":0,\"known\":0,\"faces\":[]}";
unsigned long   result_seq;

/* Seconds from a monotonic clock */
double now_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* libjpeg error handler: jump back into decode_jpeg instead of exiting the program */
void jpeg_fail(j_common_ptr cinfo) {
    longjmp(reinterpret_cast<JpegErr *>(cinfo->err)->jump, 1);
}

/* libjpeg message handler: stay quiet about minor corruption in webcam frames */
void jpeg_quiet(j_common_ptr) {}

/* Decode a JPEG to BGR, shrinking by 2, 4 or 8 while decoding as long as the longer side stays >= min_side */
cv::Mat decode_jpeg(const unsigned char *data, size_t len, int min_side) {
    jpeg_decompress_struct cinfo;
    JpegErr err;
    cinfo.err = jpeg_std_error(&err.pub);
    err.pub.error_exit     = jpeg_fail;
    err.pub.output_message = jpeg_quiet;
    err.buf = nullptr;
    if (setjmp(err.jump)) {
        jpeg_destroy_decompress(&cinfo);
        free(err.buf);
        return cv::Mat();
    }

    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, const_cast<unsigned char *>(data), len);
    jpeg_read_header(&cinfo, TRUE);

    unsigned longer = std::max(cinfo.image_width, cinfo.image_height);
    cinfo.scale_num   = 1;
    cinfo.scale_denom = 1;
    for (unsigned d = 8; d > 1; d /= 2)
        if (longer / d >= (unsigned)min_side) { cinfo.scale_denom = d; break; }
    cinfo.out_color_space = JCS_EXT_BGR;
    cinfo.dct_method      = JDCT_IFAST;
    jpeg_start_decompress(&cinfo);

    int w = cinfo.output_width, h = cinfo.output_height;
    err.buf = static_cast<unsigned char *>(malloc((size_t)w * h * 3));
    if (!err.buf) { jpeg_destroy_decompress(&cinfo); return cv::Mat(); }
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row = err.buf + (size_t)cinfo.output_scanline * w * 3;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    cv::Mat img = cv::Mat(h, w, CV_8UC3, err.buf).clone();
    free(err.buf);
    return img;
}

/* Read a whole file into memory; empty on error */
std::vector<unsigned char> read_file(const fs::path &path) {
    std::vector<unsigned char> data;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return data;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    if (size > 0) {
        data.resize(size);
        if (fread(data.data(), 1, size, f) != (size_t)size) data.clear();
    }
    fclose(f);
    return data;
}

/* Person name from a photo file name: "anna_2.jpg" -> "anna" */
std::string person_name(const fs::path &path) {
    std::string s = path.stem().string();
    size_t u = s.find_last_of('_');
    if (u != std::string::npos && u + 1 < s.size() &&
        s.find_first_not_of("0123456789", u + 1) == std::string::npos)
        s.erase(u);
    if (s.size() > MAX_NAME) s.resize(MAX_NAME);
    return s;
}

/* Load a .tflite model into an interpreter running on one thread (XNNPACK is applied on first allocation) */
bool load_net(Net &net, const std::string &path) {
    net.file = read_file(path);
    if (net.file.empty()) return false;
    net.model = tflite::FlatBufferModel::BuildFromBuffer(reinterpret_cast<const char *>(net.file.data()),
                                                         net.file.size());
    if (!net.model) return false;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*net.model, resolver)(&net.it, 1);
    return net.it != nullptr;
}

/* Shape of a tensor as text, e.g. "1x112x112x3" */
std::string shape_of(const TfLiteTensor *t) {
    std::string s;
    for (int i = 0; i < t->dims->size; i++) s += (i ? "x" : "") + std::to_string(t->dims->data[i]);
    return s;
}

/* Set up YuNet for a detect_w x detect_h input. The TFLite conversion has its 640x640 training size
   baked into two upsample sizes and the output reshapes, so rewrite those constants (they live in
   our copy of the file) before the first allocation, when XNNPACK reads them. */
bool setup_detector() {
    tflite::Interpreter &it = *detector.it;
    const TfLiteTensor *in = it.input_tensor(0);
    if (in->type != kTfLiteFloat32 || in->dims->size != 4 || in->dims->data[3] != 3) {
        fprintf(stderr, "Vision: unexpected detector input %s\n", shape_of(in).c_str());
        return false;
    }
    int side = in->dims->data[1];

    for (int n : it.execution_plan()) {
        const auto *nr = it.node_and_registration(n);
        int op = nr->second.builtin_code;
        if (op != kTfLiteBuiltinResizeNearestNeighbor && op != kTfLiteBuiltinReshape) continue;
        TfLiteTensor *shape = it.tensor(nr->first.inputs->data[1]);
        if (shape->allocation_type != kTfLiteMmapRo || shape->type != kTfLiteInt32) continue;
        int32_t *v = shape->data.i32;
        if (op == kTfLiteBuiltinResizeNearestNeighbor) {           /* [height, width] */
            int stride = side / v[0];
            v[0] = detect_h / stride;
            v[1] = detect_w / stride;
        } else if (shape->dims->data[0] == 3) {                    /* [1, cells, channels] */
            int stride = (int)std::lround(side / std::sqrt((double)v[1]));
            if (stride > 0 && side % stride == 0 && (side / stride) * (side / stride) == v[1])
                v[1] = (detect_h / stride) * (detect_w / stride);
        }
    }
    if (it.ResizeInputTensor(it.inputs()[0], { 1, detect_h, detect_w, 3 }) != kTfLiteOk ||
        it.AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "Vision: cannot run the detector at %dx%d\n", detect_w, detect_h);
        return false;
    }

    for (int h = 0; h < 4; h++)
        for (int s = 0; s < 3; s++) {
            std::string name = std::string(HEADS[h]) + "_" + std::to_string(STRIDES[s]);
            heads[h][s] = -1;
            for (int t : it.outputs())
                if (name == it.tensor(t)->name) heads[h][s] = t;
            if (heads[h][s] < 0) {
                fprintf(stderr, "Vision: detector has no output %s\n", name.c_str());
                return false;
            }
        }
    return true;
}

/* Set up SFace: one 112x112 RGB face in, one feature vector out */
bool setup_recognizer() {
    tflite::Interpreter &it = *recognizer.it;
    if (it.AllocateTensors() != kTfLiteOk) return false;
    const TfLiteTensor *in = it.input_tensor(0), *out = it.output_tensor(0);
    if (in->type != kTfLiteFloat32 || shape_of(in) != "1x112x112x3" || out->type != kTfLiteFloat32) {
        fprintf(stderr, "Vision: unexpected recognizer input %s\n", shape_of(in).c_str());
        return false;
    }
    return true;
}

/* Overlap of two boxes as intersection over union, 0..1 */
float iou(const cv::Rect2f &a, const cv::Rect2f &b) {
    float inter = (a & b).area();
    float uni   = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0;
}

/* Run YuNet on img (any size; scaled into the detector input, padded right/bottom).
   Each result row is x, y, w, h, five landmark (x, y) pairs, score, in img pixels. */
cv::Mat detect_faces(const cv::Mat &img) {
    tflite::Interpreter &it = *detector.it;
    double s = std::min((double)detect_w / img.cols, (double)detect_h / img.rows);
    cv::Mat input(detect_h, detect_w, CV_32FC3, it.typed_input_tensor<float>(0));
    input.setTo(cv::Scalar::all(0));
    cv::Mat scaled;
    cv::resize(img, scaled, cv::Size(), s, s, s < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
    scaled.convertTo(input(cv::Rect(0, 0, std::min(scaled.cols, detect_w), std::min(scaled.rows, detect_h))), CV_32F);
    if (it.Invoke() != kTfLiteOk) return cv::Mat();

    /* Decode each stride's grid the same way as OpenCV's FaceDetectorYN */
    std::vector<cv::Mat> found;
    for (int k = 0; k < 3; k++) {
        int stride = STRIDES[k], cols = detect_w / stride, rows = detect_h / stride;
        const float *cls  = it.tensor(heads[0][k])->data.f;
        const float *obj  = it.tensor(heads[1][k])->data.f;
        const float *bbox = it.tensor(heads[2][k])->data.f;
        const float *kps  = it.tensor(heads[3][k])->data.f;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                int i = r * cols + c;
                float score = std::sqrt(std::clamp(cls[i], 0.f, 1.f) * std::clamp(obj[i], 0.f, 1.f));
                if (score < score_threshold) continue;
                cv::Mat row(1, 15, CV_32F);
                float *f = row.ptr<float>();
                float cx = (c + bbox[i * 4 + 0]) * stride, cy = (r + bbox[i * 4 + 1]) * stride;
                float w = std::exp(bbox[i * 4 + 2]) * stride, h = std::exp(bbox[i * 4 + 3]) * stride;
                f[0] = cx - w / 2; f[1] = cy - h / 2; f[2] = w; f[3] = h;
                for (int n = 0; n < 5; n++) {
                    f[4 + 2 * n]     = (kps[i * 10 + 2 * n]     + c) * stride;
                    f[4 + 2 * n + 1] = (kps[i * 10 + 2 * n + 1] + r) * stride;
                }
                f[14] = score;
                row.colRange(0, 14) /= s;
                found.push_back(row);
            }
    }

    /* Keep the best-scoring box of each overlapping group */
    std::sort(found.begin(), found.end(),
              [](const cv::Mat &a, const cv::Mat &b) { return a.at<float>(14) > b.at<float>(14); });
    cv::Mat faces;
    std::vector<cv::Rect2f> kept;
    for (const auto &row : found) {
        cv::Rect2f box(row.at<float>(0), row.at<float>(1), row.at<float>(2), row.at<float>(3));
        bool overlaps = false;
        for (const auto &k : kept)
            if (iou(box, k) > NMS_THRESHOLD) { overlaps = true; break; }
        if (overlaps) continue;
        kept.push_back(box);
        faces.push_back(row);
        if (faces.rows == MAX_FACES) break;
    }
    return faces;
}

/* Similarity transform (rotation, uniform scale, shift) that best maps the five landmarks onto
   SFace's 112x112 template, by Umeyama's least-squares method, as OpenCV's alignCrop does */
cv::Mat align_matrix(const cv::Mat &row) {
    static const cv::Point2d dst[5] = { { 38.2946, 51.6963 }, { 73.5318, 51.5014 }, { 56.0252, 71.7366 },
                                        { 41.5493, 92.3655 }, { 70.7299, 92.2041 } };
    cv::Point2d src[5], sm, dm;
    for (int i = 0; i < 5; i++) {
        src[i] = cv::Point2d(row.at<float>(4 + 2 * i), row.at<float>(5 + 2 * i));
        sm += src[i] / 5;
        dm += dst[i] / 5;
    }
    cv::Matx22d cov = cv::Matx22d::zeros();
    double var = 0;
    for (int i = 0; i < 5; i++) {
        cv::Point2d a = src[i] - sm, b = dst[i] - dm;
        cov += cv::Matx22d(b.x * a.x, b.x * a.y, b.y * a.x, b.y * a.y) * 0.2;
        var += a.dot(a) / 5;
    }
    cv::Matx21d w;
    cv::Matx22d u, vt;
    cv::SVD::compute(cov, w, u, vt);
    cv::Matx22d d(1, 0, 0, cv::determinant(cov) < 0 ? -1 : 1);
    cv::Matx22d rot = u * d * vt;
    double scale = var > 0 ? (w(0) + d(1, 1) * w(1)) / var : 1;
    cv::Point2d t = dm - scale * cv::Point2d(rot(0, 0) * sm.x + rot(0, 1) * sm.y, rot(1, 0) * sm.x + rot(1, 1) * sm.y);
    return (cv::Mat_<double>(2, 3) << scale * rot(0, 0), scale * rot(0, 1), t.x,
                                      scale * rot(1, 0), scale * rot(1, 1), t.y);
}

/* SFace feature vector (unit length) of one face, aligned using its detection row */
cv::Mat face_feature(const cv::Mat &img, const cv::Mat &row) {
    tflite::Interpreter &it = *recognizer.it;
    cv::Mat aligned, rgb;
    cv::warpAffine(img, aligned, align_matrix(row), cv::Size(112, 112), cv::INTER_LINEAR);
    cv::cvtColor(aligned, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(cv::Mat(112, 112, CV_32FC3, it.typed_input_tensor<float>(0)), CV_32F);
    if (it.Invoke() != kTfLiteOk) return cv::Mat();
    const TfLiteTensor *out = it.output_tensor(0);
    cv::Mat feat = cv::Mat(1, out->dims->data[out->dims->size - 1], CV_32F, out->data.f).clone();
    double n = cv::norm(feat);
    return n > 0 ? feat / n : cv::Mat();
}

/* Feature of the largest face in a photo, trying four rotations; empty if none found */
cv::Mat photo_feature(const cv::Mat &photo) {
    const int rotations[] = { -1, cv::ROTATE_90_CLOCKWISE, cv::ROTATE_90_COUNTERCLOCKWISE, cv::ROTATE_180 };
    for (int rot : rotations) {
        cv::Mat img;
        if (rot < 0) img = photo;
        else cv::rotate(photo, img, rot);
        cv::Mat faces = detect_faces(img);
        if (faces.rows == 0) continue;
        int best = 0;
        for (int i = 1; i < faces.rows; i++)
            if (faces.at<float>(i, 2) * faces.at<float>(i, 3) >
                faces.at<float>(best, 2) * faces.at<float>(best, 3))
                best = i;
        return face_feature(img, faces.row(best));
    }
    return cv::Mat();
}

/* Load every JPEG in faces_dir as a known person */
void load_known_faces() {
    std::error_code ec;
    if (!fs::is_directory(faces_dir, ec)) {
        printf("Vision: folder %s not found, detection only\n", faces_dir.c_str());
        return;
    }

    std::vector<fs::path> files;
    for (const auto &e : fs::directory_iterator(faces_dir, ec)) {
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (e.is_regular_file(ec) && (ext == ".jpg" || ext == ".jpeg"))
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());

    for (const auto &path : files) {
        try {
            std::vector<unsigned char> data = read_file(path);
            cv::Mat photo = data.empty() ? cv::Mat() : decode_jpeg(data.data(), data.size(), 640);
            if (photo.empty()) { printf("Vision: cannot read %s\n", path.c_str()); continue; }
            cv::Mat feat = photo_feature(photo);
            if (feat.empty()) { printf("Vision: no face found in %s\n", path.c_str()); continue; }
            known.push_back({ person_name(path), feat });
            printf("Vision: %s <- %s\n", known.back().name.c_str(), path.filename().c_str());
        } catch (const std::exception &e) {
            printf("Vision: skipped %s (%s)\n", path.c_str(), e.what());
        }
    }
    printf("Vision: %zu known face photo(s)\n", known.size());
}

/* Best cosine match among known faces; "Unknown" when below the threshold */
std::string match_face(const cv::Mat &feat, float &sim) {
    std::string name = "Unknown";
    sim = 0;
    if (feat.empty()) return name;
    for (const auto &k : known) {
        float s = (float)feat.dot(k.feature);
        if (s > sim) {
            sim = s;
            name = s >= match_threshold ? k.name : "Unknown";
        }
    }
    return name;
}

/* Escape a string for use inside a JSON string value */
std::string json_escape(const std::string &s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char)c; }
        else if (c < 0x20) { char u[8]; snprintf(u, sizeof u, "\\u%04x", c); out += u; }
        else out += (char)c;
    }
    return out;
}

/* Results as JSON; box coordinates are fractions of the frame size */
std::string to_json(const std::vector<Track> &tracks, double fps) {
    char num[160];
    snprintf(num, sizeof num, "{\"fps\":%.1f,\"known\":%zu,\"faces\":[", fps, known.size());
    std::string json = num;
    for (size_t i = 0; i < tracks.size(); i++) {
        const Track &t = tracks[i];
        snprintf(num, sizeof num,
                 "%s{\"x\":%.4f,\"y\":%.4f,\"w\":%.4f,\"h\":%.4f,\"score\":%.2f,\"sim\":%.2f,\"name\":\"",
                 i ? "," : "", t.box.x, t.box.y, t.box.width, t.box.height, t.score, t.sim);
        json += num;
        json += json_escape(t.name);
        json += "\"}";
    }
    json += "]}";
    return json;
}

/* Detection thread: decode the newest frame, find and name faces, publish the JSON result */
void *vision_loop(void *) {
    setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10);

    unsigned char *jpeg = nullptr;
    size_t cap = 0;
    unsigned long seq = 0;
    std::vector<Track> tracks;
    double last = now_sec(), fps = 0;

    for (;;) {
        size_t len = camera_wait_frame(&seq, &jpeg, &cap);
        if (len == 0) continue;

        std::vector<Track> found;
        try {
            cv::Mat frame = decode_jpeg(jpeg, len, FRAME_SIDE);
            if (frame.empty()) continue;
            cv::Mat faces = detect_faces(frame);

            for (int i = 0; i < faces.rows; i++) {
                cv::Mat row = faces.row(i);
                Track t;
                t.box = cv::Rect2f(row.at<float>(0) / frame.cols, row.at<float>(1) / frame.rows,
                                   row.at<float>(2) / frame.cols, row.at<float>(3) / frame.rows);
                t.score = row.at<float>(14);
                t.sim = 0;
                t.recog_at = 0;

                const Track *prev = nullptr;
                float best = 0.3f;
                for (const auto &p : tracks) {
                    float o = iou(p.box, t.box);
                    if (o > best) { best = o; prev = &p; }
                }

                if (known.empty()) {
                    t.name = "";
                } else if (prev && now_sec() - prev->recog_at < recog_interval) {
                    t.name = prev->name;
                    t.sim = prev->sim;
                    t.recog_at = prev->recog_at;
                } else {
                    t.name = match_face(face_feature(frame, row), t.sim);
                    t.recog_at = now_sec();
                }
                found.push_back(t);
            }
        } catch (const std::exception &e) {
            fprintf(stderr, "Vision: %s\n", e.what());
            continue;
        }
        tracks = found;

        double now = now_sec(), dt = now - last;
        last = now;
        if (dt > 0) fps = fps > 0 ? 0.8 * fps + 0.2 / dt : 1.0 / dt;

        std::string json = to_json(tracks, fps);
        pthread_mutex_lock(&lock);
        result = json;
        result_seq++;
        pthread_cond_broadcast(&updated);
        pthread_mutex_unlock(&lock);
    }
    return nullptr;
}

}

/* Load the models and known faces, then start the detection thread */
extern "C" int vision_start(const struct vision_config *cfg) {
    detect_model    = cfg->detect_model;
    recog_model     = cfg->recog_model;
    faces_dir       = cfg->faces_dir;
    detect_w        = cfg->detect_width;
    detect_h        = cfg->detect_height;
    score_threshold = cfg->score_threshold;
    match_threshold = cfg->match_threshold;
    recog_interval  = cfg->recog_interval;

    if (detect_w <= 0 || detect_h <= 0 || detect_w % 32 || detect_h % 32) {
        fprintf(stderr, "Vision: detect size %dx%d must be multiples of 32\n", detect_w, detect_h);
        return -1;
    }
    if (!load_net(detector, detect_model)) {
        fprintf(stderr, "Vision: cannot load %s, face detection off\n", detect_model.c_str());
        return -1;
    }
    if (!setup_detector()) return -1;

    if (!load_net(recognizer, recog_model)) {
        fprintf(stderr, "Vision: cannot load %s, detection only\n", recog_model.c_str());
    } else if (!setup_recognizer()) {
        fprintf(stderr, "Vision: cannot use %s, detection only\n", recog_model.c_str());
    } else {
        have_recognizer = true;
    }

    /* One timed run of each model, so the speed on this board shows in the log */
    cv::Mat probe(detect_h, detect_w, CV_8UC3, cv::Scalar(128, 128, 128));
    double t0 = now_sec();
    detect_faces(probe);
    double t1 = now_sec();
    printf("Vision: detect %.2f s", t1 - t0);
    if (have_recognizer) {
        cv::Mat row = cv::Mat::zeros(1, 15, CV_32F);
        const float pts[10] = { 38, 52, 74, 52, 56, 72, 42, 92, 71, 92 };
        for (int i = 0; i < 10; i++) row.at<float>(4 + i) = pts[i];
        face_feature(probe, row);
        printf(", recognize %.2f s", now_sec() - t1);
    }
    printf("\n");

    if (have_recognizer) load_known_faces();

    pthread_t t;
    if (pthread_create(&t, nullptr, vision_loop, nullptr) != 0) {
        fprintf(stderr, "Vision: pthread_create failed\n");
        return -1;
    }
    pthread_detach(t);
    printf("Vision: running, detecting at %dx%d\n", detect_w, detect_h);
    return 0;
}

/* Wait up to 2 s for a result newer than *seq and copy its JSON into buf */
extern "C" size_t vision_wait_result(unsigned long *seq, char *buf, size_t size) {
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += 2;

    pthread_mutex_lock(&lock);
    while (result_seq == *seq) {
        if (pthread_cond_timedwait(&updated, &lock, &deadline) == ETIMEDOUT) {
            pthread_mutex_unlock(&lock);
            return 0;
        }
    }
    *seq = result_seq;
    size_t len = result.size();
    if (len >= size) len = 0;
    else memcpy(buf, result.c_str(), len + 1);
    pthread_mutex_unlock(&lock);
    return len;
}
