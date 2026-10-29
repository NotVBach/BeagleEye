#!/usr/bin/env bash
# Fetch YuNet (prebuilt TFLite) and SFace (ONNX -> TFLite via onnx2tf) into models/.
set -euo pipefail

OUT="$(cd "$(dirname "$0")/.." && pwd)/models"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"
cd "$WORK"

# YuNet: PINTO's float32 TFLite matches OpenCV's reference.
if [ ! -f "$OUT/face_detection_yunet_2023mar.tflite" ]; then
    curl -fsSL https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/387_YuNetV2/resources.tar.gz \
        | tar -xz saved_model/face_detection_yunet_2023mar_float32.tflite
    mv saved_model/face_detection_yunet_2023mar_float32.tflite "$OUT/face_detection_yunet_2023mar.tflite"
fi

# SFace: PINTO's TFLite is broken, so convert the official ONNX ourselves.
if [ ! -f "$OUT/face_recognition_sface_2021dec.tflite" ]; then
    curl -fsSLo sface.onnx \
        https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
    # TF 2.17 needs Python <= 3.12; uv fetches its own interpreter (standalone binary, no system install).
    UV="$(command -v uv || true)"
    if [ -z "$UV" ]; then
        curl -fsSL https://github.com/astral-sh/uv/releases/download/0.5.11/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz
        UV="$WORK/uv-x86_64-unknown-linux-gnu/uv"
    fi
    "$UV" venv -q -p 3.11 venv
    VIRTUAL_ENV="$WORK/venv" "$UV" pip install -q onnx2tf==1.27.10 tensorflow==2.17.0 tf_keras==2.17.0 \
        onnx==1.16.1 onnxruntime==1.18.1 onnx_graphsurgeon onnxsim sng4onnx psutil ai_edge_litert
    # onnx2tf's calibration-data download link is dead; any local file with this name satisfies it.
    venv/bin/python -c "import numpy as np; np.save('calibration_image_sample_data_20x128x128x3_float32.npy', np.random.rand(20,128,128,3).astype('float32'))"
    venv/bin/onnx2tf -i sface.onnx -o tf_out   # rewrites sface.onnx in place; it's a temp copy
    mv tf_out/sface_float32.tflite "$OUT/face_recognition_sface_2021dec.tflite"
fi

ls -l "$OUT"
