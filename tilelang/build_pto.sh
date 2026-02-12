#!/bin/bash
# Build script for PTO kernel compilation using bisheng compiler

# ============================================================================
# Configuration
# ============================================================================

# Source file (input .cpp file)
# Usage: ./build_pto.sh <source_file> [platform]
# Default: kernel.cpp, A2
SRC_FILE="${1:-kernel.cpp}"
PLATFORM="${2:-A2}"

# Output library file
# LIB_FILE="${SRC_FILE%.cpp}.so"
LIB_FILE="pto.so"
# Platform-specific settings
case "$PLATFORM" in
    A5)
        CCE_ARCH="dav-c310"
        MEMORY_DEF="REGISTER_BASE"
        ;;
    A2|*)
        CCE_ARCH="dav-c220"
        MEMORY_DEF="MEMORY_BASE"
        ;;
esac

# Environment variables (can be overridden)
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
TL_ROOT="${TL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
TILELANG_TEMPLATE_PATH="${TILELANG_TEMPLATE_PATH:-$TL_ROOT/src}"

# ============================================================================
# Compilation
# ============================================================================

echo "=========================================="
echo "PTO Kernel Build Script"
echo "=========================================="
echo "Source file:    $SRC_FILE"
echo "Output library: $LIB_FILE"
echo "Platform:       $PLATFORM"
echo "CCE Arch:       $CCE_ARCH"
echo "Memory mode:    $MEMORY_DEF"
echo "ASCEND_HOME:    $ASCEND_HOME_PATH"
echo "TL_ROOT:        $TL_ROOT"
echo "=========================================="

bisheng \
    --cce-aicore-arch="$CCE_ARCH" \
    -D_DEBUG --cce-enable-print \
    -D"$MEMORY_DEF" \
    -O2 \
    -std=gnu++17 \
    -xcce \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -DL2_CACHE_HINT \
    -I"../../src/" \
    -I"$TL_ROOT/3rdparty/pto-isa/include" \
    -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/include/experiment/msprof" \
    -I"$ASCEND_HOME_PATH/include/experiment/runtime" \
    -I"/usr/local/Ascend/driver/kernel/inc" \
    -I"$ASCEND_HOME_PATH/pkg_inc" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
    -I"$ASCEND_HOME_PATH/pkg_inc/profiling" \
    -L"$ASCEND_HOME_PATH/lib64" \
    -I"$TILELANG_TEMPLATE_PATH" \
    -Wno-macro-redefined \
    -Wno-ignored-attributes \
    -lruntime \
    -lstdc++ \
    -lascendcl \
    -lm \
    -ltiling_api \
    -lplatform \
    -lc_sec \
    -ldl \
    -fPIC \
    --shared \
    "$SRC_FILE" \
    -o "$LIB_FILE"

# Check compilation result
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Build successful! Output: $LIB_FILE"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "Build failed!"
    echo "=========================================="
    exit 1
fi
