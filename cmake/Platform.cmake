# Platform.cmake - Platform detection and base configuration
#
# This module handles:
# - C++ standard configuration
# - Platform detection (ONNX Runtime vs CVI TPU)
# - T-Head RISC-V optimization detection
# - JPEG decode method selection

# =============================================================================
# C++ Standard Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress CMAKE_TOOLCHAIN_FILE unused warning
if(CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# =============================================================================
# Platform Detection
# =============================================================================
# When using toolchain file (cmake/toolchain-sg200x.cmake),
# USE_ONNX_RUNTIME and USE_CVI_TPU are set by the toolchain.
# Otherwise, default to CPU-only build with ONNX Runtime.

if(NOT DEFINED USE_ONNX_RUNTIME)
    set(USE_ONNX_RUNTIME ON)
    message(STATUS "ONNX Runtime: ENABLED (CPU inference, default build)")
endif()

if(USE_ONNX_RUNTIME)
    message(STATUS "Build mode: CPU-only (ONNX Runtime)")
    set(SG200X_BUILD OFF)
else()
    message(STATUS "Build mode: SG200X TPU (CVI Runtime)")
    set(SG200X_BUILD ON)

    # TPU paths should be set by toolchain file
    if(NOT DEFINED TPU_SDK_ROOT)
        message(FATAL_ERROR
            "TPU SDK paths not configured.\n"
            "Please use: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake .."
        )
    endif()
endif()

# =============================================================================
# T-Head RISC-V Optimizations (RVV)
# =============================================================================
option(ENABLE_RV_THEAD "Enable T-Head RISC-V vector optimizations" ON)

if(ENABLE_RV_THEAD AND SG200X_BUILD)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-march=rv64gcv0p7_zfh_xthead")

    check_cxx_source_compiles("
        #include <riscv_vector.h>
        int main() {
            vfloat32m2_t v = vfmv_v_f_f32m2(0.0f, 4);
            return 0;
        }
    " HAVE_RV_THEAD)

    if(HAVE_RV_THEAD)
        message(STATUS "T-Head RISC-V optimizations: ENABLED")
        add_compile_definitions(ENABLE_RV_THEAD)
        set(RV_THEAD_ENABLED ON)
    else()
        message(WARNING "Compiler does not support RVV, disabling T-Head optimizations")
        set(ENABLE_RV_THEAD OFF)
        set(RV_THEAD_ENABLED OFF)
    endif()
else()
    message(STATUS "T-Head RISC-V optimizations: DISABLED (CPU fallback)")
    set(RV_THEAD_ENABLED OFF)
endif()

# =============================================================================
# JPEG Decode Method Selection
# =============================================================================
option(USE_VDEC_DECODE "Use hardware VDEC for JPEG decode (requires VB pool)" OFF)

if(USE_VDEC_DECODE)
    message(STATUS "JPEG decode: Hardware VDEC (requires VB pool)")
    add_compile_definitions(USE_VDEC_DECODE)
else()
    message(STATUS "JPEG decode: Software (OpenCV, no VB pool dependency)")
endif()

# =============================================================================
# Camera Support Option
# =============================================================================
option(ENABLE_CVI_CAMERA "Enable Camera support with ISP" OFF)

if(SG200X_BUILD)
    if(ENABLE_CVI_CAMERA)
        message(STATUS "Camera support: ENABLED (ISP + VI)")
        add_compile_definitions(USE_CVI_CAMERA)
    else()
        message(STATUS "Camera support: DISABLED (VPSS only)")
    endif()
endif()
