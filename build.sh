#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Tried in order; the first reachable one is used.
HOSTS=("debian@192.168.7.2" "debian@192.168.1.9")
BBB_DIR="BeagleEye"
BUILD_DIR="build"
TOOLCHAIN="cmake/toolchain.cmake"

step() { printf '\n==> %s\n' "$*"; }
die()  { printf '\nError: %s\n' "$*" >&2; exit 1; }

step "Checking tools"
for tool in cmake arm-linux-gnueabihf-g++-12 ssh scp; do
    command -v "$tool" >/dev/null || die "$tool not found. Install: sudo apt install cmake g++-12-arm-linux-gnueabihf openssh-client"
done
[ -f "$TOOLCHAIN" ] || die "$TOOLCHAIN not found"
ls models/*.tflite >/dev/null 2>&1 || die "no models in models/. Run: cmake/get-models.sh"
echo "OK"

step "Building"
export CMAKE_POLICY_VERSION_MINIMUM=3.5  # TFLite deps declare pre-3.5 minimums that CMake 4 rejects
cmake -S . -B "$BUILD_DIR" -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" >/dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)"

step "Finding a reachable BeagleBone"
HOST=""
for h in "${HOSTS[@]}"; do
    echo "Trying $h ..."
    if ssh -o ConnectTimeout=5 "$h" true; then
        HOST="$h"
        break
    fi
    echo "  $h not reachable"
done
[ -n "$HOST" ] || die "cannot reach any host (${HOSTS[*]}). Is the board connected and powered?"
echo "OK, using $HOST"

step "Deploying to $HOST:~/$BBB_DIR"
ssh "$HOST" "mkdir -p $BBB_DIR/server $BBB_DIR/www $BBB_DIR/models"
scp -q "$BUILD_DIR/server" "$HOST:$BBB_DIR/server/server.new"
# Models are large; copy only the ones the board doesn't have yet.
for m in models/*.tflite; do
    ssh "$HOST" "test -f $BBB_DIR/$m" || scp -q "$m" "$HOST:$BBB_DIR/$m"
done
scp -q www/index.html "$HOST:$BBB_DIR/www/index.html"
ssh "$HOST" "mv -f $BBB_DIR/server/server.new $BBB_DIR/server/server"
echo "OK"

step "Running on $HOST (Ctrl+C to stop)"
echo "Open http://${HOST#*@}:8080/"
ssh -t "$HOST" "pkill -f '^\./server/server' || true; cd $BBB_DIR && exec ./server/server"