# Dependencies.cmake - External dependency configuration
#
# This module handles:
# - OpenCV configuration
# - ONNX Runtime configuration (CPU builds)
# - ALSA configuration (SG200X builds)

# =============================================================================
# OpenCV
# =============================================================================
# Temporarily suppress warnings from OpenCV's config file (SG200X SDK issue)
set(_SAVED_CMAKE_WARN_DEPRECATED ${CMAKE_WARN_DEPRECATED})
set(CMAKE_WARN_DEPRECATED OFF)

find_package(OpenCV 4.5.0 REQUIRED)

set(CMAKE_WARN_DEPRECATED ${_SAVED_CMAKE_WARN_DEPRECATED})
unset(_SAVED_CMAKE_WARN_DEPRECATED)

# Fix OpenCV include path issue on SG200X SDK
if(SG200X_BUILD)
    set(_fixed_opencv_includes "")
    foreach(_dir ${OpenCV_INCLUDE_DIRS})
        if(EXISTS "${_dir}")
            list(APPEND _fixed_opencv_includes "${_dir}")
        else()
            message(STATUS "Skipping non-existent OpenCV include: ${_dir}")
        endif()
    endforeach()

    if(NOT "${TPU_SDK_ROOT}/include" IN_LIST _fixed_opencv_includes)
        list(APPEND _fixed_opencv_includes "${TPU_SDK_ROOT}/include")
    endif()

    set(OpenCV_INCLUDE_DIRS ${_fixed_opencv_includes})
    message(STATUS "Corrected OpenCV includes: ${OpenCV_INCLUDE_DIRS}")
endif()

# =============================================================================
# ONNX Runtime (CPU builds only)
# =============================================================================
if(USE_ONNX_RUNTIME)
    set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime-prebuilt")
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
    )

    add_compile_definitions(USE_ONNX_RUNTIME)
    message(STATUS "ONNX Runtime library configured")
endif()

# =============================================================================
# ALSA (SG200X builds only)
# =============================================================================
if(SG200X_BUILD)
    set(ALSA_SYSROOT "${SG200X_SDK_PATH}/buildroot-2021.05/output/cvitek_CV181X_musl_riscv64/host/riscv64-buildroot-linux-musl/sysroot")
    set(ALSA_INCLUDE_DIR "${ALSA_SYSROOT}/usr/include")
    set(ALSA_LIB_DIR "${ALSA_SYSROOT}/usr/lib")
    message(STATUS "ALSA audio capture: enabled (runtime library)")
    message(STATUS "ALSA sysroot: ${ALSA_SYSROOT}")

    # Mosquitto (for MQTT protocol tests)
    set(MOSQUITTO_INCLUDE_DIR "${ALSA_SYSROOT}/usr/include")
    set(MOSQUITTO_LIB_DIR "${ALSA_SYSROOT}/usr/lib")
endif()

# =============================================================================
# nlohmann/json (header-only)
# =============================================================================
set(NLOHMANN_JSON_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/nlohmann")
