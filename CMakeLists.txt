cmake_minimum_required(VERSION 3.18)
project(ModelInfer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress CMAKE_TOOLCHAIN_FILE unused warning (it's used internally by CMake)
if(CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# ==========================================================
# 0. Platform Detection and Configuration
# ==========================================================
# Note: When using toolchain file (cmake/toolchain-sg200x.cmake),
#       USE_ONNX_RUNTIME and USE_CVI_TPU are set by the toolchain.
#       Otherwise, default to CPU-only build with ONNX Runtime.

if(NOT DEFINED USE_ONNX_RUNTIME)
    # Default: CPU-only build with ONNX Runtime
    set(USE_ONNX_RUNTIME ON)
    message(STATUS "ONNX Runtime: ENABLED (CPU inference, default build)")
endif()

if(USE_ONNX_RUNTIME)
    message(STATUS "Build mode: CPU-only (ONNX Runtime)")
else()
    message(STATUS "Build mode: SG200X TPU (CVI Runtime)")

    # TPU paths should be set by toolchain file
    if(NOT DEFINED TPU_SDK_ROOT)
        message(FATAL_ERROR
            "TPU SDK paths not configured.\n"
            "Please use: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake .."
        )
    endif()
endif()

# ==========================================================
# 1. Build Lua as C++ (Critical for exception safety)
# ==========================================================
file(GLOB LUA_SRC "lua/*.c")
# Exclude standalone interpreter and compiler
list(REMOVE_ITEM LUA_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/lua/lua.c" 
    "${CMAKE_CURRENT_SOURCE_DIR}/lua/luac.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/lua/onelua.c"
)

add_library(lua STATIC ${LUA_SRC})
target_compile_options(lua PRIVATE -x c++ -O3 -Wall -DLUA_USE_POSIX)
target_include_directories(lua PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lua")

# ==========================================================
# 2. LuaIntf
# ==========================================================
# LuaIntf is header-only, but we need to make sure it finds our Lua
add_library(LuaIntf INTERFACE)
target_include_directories(LuaIntf INTERFACE 
    "${CMAKE_CURRENT_SOURCE_DIR}/lua-intf-ex/src/include"
)
target_link_libraries(LuaIntf INTERFACE lua)

# ==========================================================
# 3. Dependencies
# ==========================================================
# Temporarily suppress warnings from OpenCV's config file (SG200X SDK issue)
set(_SAVED_CMAKE_WARN_DEPRECATED ${CMAKE_WARN_DEPRECATED})
set(CMAKE_WARN_DEPRECATED OFF)

find_package(OpenCV 4.5.0 REQUIRED)

# Restore warning settings
set(CMAKE_WARN_DEPRECATED ${_SAVED_CMAKE_WARN_DEPRECATED})
unset(_SAVED_CMAKE_WARN_DEPRECATED)

# Fix OpenCV include path issue on SG200X SDK
# The SDK's OpenCVConfig.cmake references a non-existent opencv4 directory
if(NOT USE_ONNX_RUNTIME)
    # Check if OpenCV_INCLUDE_DIRS contains non-existent paths
    set(_fixed_opencv_includes "")
    foreach(_dir ${OpenCV_INCLUDE_DIRS})
        if(EXISTS "${_dir}")
            list(APPEND _fixed_opencv_includes "${_dir}")
        else()
            message(STATUS "Skipping non-existent OpenCV include: ${_dir}")
        endif()
    endforeach()

    # Ensure the actual include directory is present
    if(NOT "${TPU_SDK_ROOT}/include" IN_LIST _fixed_opencv_includes)
        list(APPEND _fixed_opencv_includes "${TPU_SDK_ROOT}/include")
    endif()

    set(OpenCV_INCLUDE_DIRS ${_fixed_opencv_includes})
    message(STATUS "Corrected OpenCV includes: ${OpenCV_INCLUDE_DIRS}")
endif()

# ONNX Runtime (only on x86_64/ARM platforms)
if(USE_ONNX_RUNTIME)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-prebuilt")
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

# ==========================================================
# 4. Main Application (Lua Runner)
# ==========================================================
# 收集模块源文件
file(GLOB MODULE_SOURCES "src/modules/*.cpp")
file(GLOB TENSOR_SOURCES "src/modules/tensor/*.cpp")
file(GLOB CV_SOURCES "src/modules/cv/*.cpp")
file(GLOB BINDING_SOURCES "src/bindings/*.cpp")
file(GLOB INFERENCE_SOURCES "src/inference/*.cpp")

# 条件排除 CVI/VPSS 后端（无 SDK 时）
if(USE_ONNX_RUNTIME)
    # CPU-only build: exclude hardware-accelerated components
    list(REMOVE_ITEM CV_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/cv/cvi_camera.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/cv/cvi_sensor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/cv/cvi_vpss_processor.cpp"
    )
    message(STATUS "CV module: CPU-only (OpenCV)")
else()
    # SG200X build: enable CVI MPI support
    # NOTE: CviCamera and CviSensor disabled by default (require ISP libraries)
    #       Enable with -DENABLE_CVI_CAMERA=ON for full Camera support
    option(ENABLE_CVI_CAMERA "Enable Camera support with ISP" OFF)

    if(NOT ENABLE_CVI_CAMERA)
        list(REMOVE_ITEM CV_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/cv/cvi_camera.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/cv/cvi_sensor.cpp"
        )
        message(STATUS "CV module: Hardware-accelerated (VPSS only, Camera disabled)")
    else()
        message(STATUS "CV module: Hardware-accelerated (Camera + ISP + VPSS enabled)")
    endif()

    add_compile_definitions(USE_CVI_MPI)
endif()

add_executable(lua_runner
    src/main.cpp
    ${MODULE_SOURCES}
    ${TENSOR_SOURCES}
    ${CV_SOURCES}
    ${BINDING_SOURCES}
    ${INFERENCE_SOURCES}
)

target_link_libraries(lua_runner PRIVATE
    LuaIntf
    lua
    ${OpenCV_LIBS}
)

# Conditionally link ONNX Runtime
if(USE_ONNX_RUNTIME)
    target_link_libraries(lua_runner PRIVATE onnxruntime)
endif()

target_include_directories(lua_runner PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/tensor"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/cv"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils"
    ${OpenCV_INCLUDE_DIRS}
)

# Conditionally include ONNX Runtime headers
if(USE_ONNX_RUNTIME)
    target_include_directories(lua_runner PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
endif()

# Add Sophgo TPU support if configured
if(NOT USE_ONNX_RUNTIME)
    target_include_directories(lua_runner PRIVATE
        ${TPU_INCLUDE_DIR}
        ${CVI_MPI_INCLUDE_DIR}
    )
    target_link_directories(lua_runner PRIVATE
        ${TPU_LIB_DIR}
        ${CVI_MPI_LIB_DIR}
    )

    # Base libraries (always needed for TPU/VPSS)
    set(CVI_BASE_LIBS
        ${TPU_LIB_DIR}/libcviruntime-static.a
        ${TPU_LIB_DIR}/libcvikernel-static.a
        ${TPU_LIB_DIR}/libcvimath-static.a
        ${TPU_LIB_DIR}/libflatbuffers.a
        ${CVI_MPI_LIB_DIR}/libsys.a
        ${CVI_MPI_LIB_DIR}/libvpss.a
        ${CVI_MPI_LIB_DIR}/libvdec.a
        ${CVI_MPI_LIB_DIR}/libvo.a
    )

    # Camera/ISP libraries (only when Camera enabled)
    if(ENABLE_CVI_CAMERA)
        # Add ISP include directory for camera headers
        target_include_directories(lua_runner PRIVATE ${CVI_ISP_INCLUDE_DIR})

        set(CVI_CAMERA_LIBS
            ${CVI_MPI_LIB_DIR}/libvi.a
            ${CVI_MPI_LIB_DIR}/libisp.a
            ${CVI_MPI_LIB_DIR}/libisp_algo.a
            ${CVI_MPI_LIB_DIR}/libae.a
            ${CVI_MPI_LIB_DIR}/libawb.a
            ${CVI_MPI_LIB_DIR}/libaf.a
            ${CVI_MPI_LIB_DIR}/libcvi_bin.a
            ${CVI_MPI_LIB_DIR}/libcvi_bin_isp.a
            ${CVI_MPI_LIB_DIR}/libsns_full.a
        )
        target_compile_definitions(lua_runner PRIVATE USE_CVI_CAMERA)

        # Use link group for proper static library resolution (ISP depends on sys)
        target_link_libraries(lua_runner PRIVATE
            -static
            ${TPU_LIB_DIR}/libcviruntime-static.a
            ${TPU_LIB_DIR}/libcvikernel-static.a
            ${TPU_LIB_DIR}/libcvimath-static.a
            ${TPU_LIB_DIR}/libflatbuffers.a
            -Wl,--start-group
            ${CVI_CAMERA_LIBS}
            ${CVI_MPI_LIB_DIR}/libsys.a
            ${CVI_MPI_LIB_DIR}/libvpss.a
            ${CVI_MPI_LIB_DIR}/libvdec.a
            ${CVI_MPI_LIB_DIR}/libvo.a
            -Wl,--end-group
            z
            m
            pthread
            dl
            atomic
        )
    else()
        set(CVI_CAMERA_LIBS "")

        target_link_libraries(lua_runner PRIVATE
            -static
            ${CVI_BASE_LIBS}
            z
            m
            pthread
            dl
            atomic
        )
    endif()
    message(STATUS "lua_runner: Sophgo TPU support linked")
endif()

# Optional: Link dl library for Linux
if(UNIX AND NOT APPLE)
    target_link_libraries(lua_runner PRIVATE dl)
endif()

# Set RPATH for ONNX Runtime (only when enabled)
if(USE_ONNX_RUNTIME)
    set_target_properties(lua_runner PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
        INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    )
endif()

# ==========================================================
# 5. Pure C++ Inference App (ONNX Runtime or CVI Runtime)
# ==========================================================
add_executable(cpp_infer
    src/cpp_main.cpp
    ${INFERENCE_SOURCES}
)

target_link_libraries(cpp_infer PRIVATE
    ${OpenCV_LIBS}
)

target_include_directories(cpp_infer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
)

if(USE_ONNX_RUNTIME)
    target_link_libraries(cpp_infer PRIVATE onnxruntime)
    target_include_directories(cpp_infer PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})

    set_target_properties(cpp_infer PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
        INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    )
else()
    target_sources(cpp_infer PRIVATE
        src/modules/tensor/cvi_tpu_memory.cpp
    )
    target_include_directories(cpp_infer PRIVATE ${TPU_INCLUDE_DIR})
    target_link_directories(cpp_infer PRIVATE ${TPU_LIB_DIR})
    target_link_libraries(cpp_infer PRIVATE
        -static
        ${TPU_LIB_DIR}/libcviruntime-static.a
        ${TPU_LIB_DIR}/libcvikernel-static.a
        ${TPU_LIB_DIR}/libcvimath-static.a
        ${TPU_LIB_DIR}/libflatbuffers.a
        z
        m
        pthread
        dl
        atomic
    )
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(cpp_infer PRIVATE dl)
endif()

message(STATUS "cpp_infer: inference benchmark enabled")

# ==========================================================
# 6. Test Programs for SG200X
# ==========================================================
if(NOT USE_ONNX_RUNTIME)
    # Test 1: TPU Memory Management
    add_executable(test_tpu_memory
        tests/test_tpu_memory.cpp
        src/modules/tensor/device_buffer.cpp
        src/modules/tensor/cpu_memory.cpp
        src/modules/tensor/cvi_tpu_memory.cpp
    )

    target_include_directories(test_tpu_memory PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/tensor"
        ${TPU_INCLUDE_DIR}
    )

    target_link_directories(test_tpu_memory PRIVATE
        ${TPU_LIB_DIR}
    )

    # Static linking for embedded deployment
    target_link_libraries(test_tpu_memory PRIVATE
        -static
        ${TPU_LIB_DIR}/libcviruntime-static.a
        ${TPU_LIB_DIR}/libcvikernel-static.a
        ${TPU_LIB_DIR}/libcvimath-static.a
        ${TPU_LIB_DIR}/libflatbuffers.a
        z
        m
        pthread
        dl
    )

    message(STATUS "test_tpu_memory: TPU memory test enabled (static build)")

    # Test 2: CV Module Comprehensive Test
    add_executable(test_cv_func
        tests/test_cv_func.cpp
        tests/test_common.cpp
        tests/test_frame.cpp
        tests/test_opencv_proc.cpp
        tests/test_cv_helpers.cpp
        tests/test_vpss_proc.cpp
        tests/test_benchmark.cpp
        tests/test_real_image.cpp
        tests/test_vpss_performance.cpp
        tests/test_ion_loader.cpp
        tests/test_camera_capture.cpp
        src/modules/cv/cvi_vpss_processor.cpp
        src/modules/cv/ion_image_loader.cpp
        src/modules/cv/hw_jpeg_decoder.cpp
        src/modules/cv/opencv_processor.cpp
        src/modules/cv/cvi_frame.cpp
        src/modules/cv/cv_helpers.cpp
        src/modules/cv/cv_types.cpp
        src/modules/cv/device_buffer.cpp
        src/modules/cv/image.cpp
        src/modules/cv/mmf_context.cpp
        ${TENSOR_SOURCES}
    )

    target_include_directories(test_cv_func PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/modules"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/cv"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/tensor"
        "${CMAKE_CURRENT_SOURCE_DIR}/lua-intf-ex/src/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/lua"
        ${OpenCV_INCLUDE_DIRS}
        ${TPU_INCLUDE_DIR}
        ${CVI_MPI_INCLUDE_DIR}
    )

    target_link_directories(test_cv_func PRIVATE
        ${TPU_LIB_DIR}
        ${CVI_MPI_LIB_DIR}
    )

    target_link_libraries(test_cv_func PRIVATE
        -static
        lua
        ${TPU_LIB_DIR}/libcviruntime-static.a
        ${TPU_LIB_DIR}/libcvikernel-static.a
        ${TPU_LIB_DIR}/libcvimath-static.a
        ${TPU_LIB_DIR}/libflatbuffers.a
        -Wl,--start-group
        ${CVI_MPI_LIB_DIR}/libsys.a
        ${CVI_MPI_LIB_DIR}/libvpss.a
        ${CVI_MPI_LIB_DIR}/libvdec.a
        ${CVI_MPI_LIB_DIR}/libvo.a
        -Wl,--end-group
        ${OpenCV_LIBS}
        z
        m
        pthread
        dl
        atomic
    )

    message(STATUS "test_cv_func: CV module comprehensive test enabled (static build)")

    # Add camera sources and ISP libraries when ENABLE_CVI_CAMERA is ON
    if(ENABLE_CVI_CAMERA)
        target_sources(test_cv_func PRIVATE
            src/modules/cv/cvi_camera.cpp
            src/modules/cv/cvi_sensor.cpp
        )

        target_include_directories(test_cv_func PRIVATE
            ${CVI_ISP_INCLUDE_DIR}
        )

        # Re-link with ISP libraries inside link group for proper static resolution
        target_link_libraries(test_cv_func PRIVATE
            -Wl,--start-group
            ${CVI_MPI_LIB_DIR}/libvi.a
            ${CVI_MPI_LIB_DIR}/libisp.a
            ${CVI_MPI_LIB_DIR}/libisp_algo.a
            ${CVI_MPI_LIB_DIR}/libae.a
            ${CVI_MPI_LIB_DIR}/libawb.a
            ${CVI_MPI_LIB_DIR}/libaf.a
            ${CVI_MPI_LIB_DIR}/libcvi_bin.a
            ${CVI_MPI_LIB_DIR}/libcvi_bin_isp.a
            ${CVI_MPI_LIB_DIR}/libsns_full.a
            ${CVI_MPI_LIB_DIR}/libsys.a
            -Wl,--end-group
        )

        target_compile_definitions(test_cv_func PRIVATE USE_CVI_CAMERA)

        message(STATUS "test_cv_func: Camera capture tests enabled (ISP libraries linked)")
    endif()

    # Test 3: VDEC Standalone Test (isolated JPEG decoder test)
    # Static linking for embedded deployment
    add_executable(test_vdec_standalone
        tests/test_vdec_standalone.cpp
    )

    target_include_directories(test_vdec_standalone PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${TPU_INCLUDE_DIR}
        ${CVI_MPI_INCLUDE_DIR}
        ${CVI_MPI_INCLUDE_DIR}/common
        ${CVI_MPI_INCLUDE_DIR}/linux
    )

    target_link_directories(test_vdec_standalone PRIVATE
        ${TPU_LIB_DIR}
        ${CVI_MPI_LIB_DIR}
        ${CVI_MPI_LIB_DIR}/3rd
    )

    # Static linking for VDEC test
    target_link_libraries(test_vdec_standalone PRIVATE
        -static
        ${CVI_MPI_LIB_DIR}/libvdec.a
        ${CVI_MPI_LIB_DIR}/libmisc.a
        ${CVI_MPI_LIB_DIR}/libsys.a
        ${CVI_MPI_LIB_DIR}/libvi.a
        ${CVI_MPI_LIB_DIR}/libvpss.a
        ${CVI_MPI_LIB_DIR}/libvo.a
        ${CVI_MPI_LIB_DIR}/libgdc.a
        ${CVI_MPI_LIB_DIR}/libvenc.a
        ${TPU_LIB_DIR}/libcviruntime-static.a
        ${TPU_LIB_DIR}/libcvikernel-static.a
        ${TPU_LIB_DIR}/libcvimath-static.a
        ${TPU_LIB_DIR}/libflatbuffers.a
        ${CVI_MPI_LIB_DIR}/3rd/libini.a
        z
        m
        pthread
        dl
        atomic
    )

    target_compile_definitions(test_vdec_standalone PRIVATE USE_CVI_MPI)

    message(STATUS "test_vdec_standalone: VDEC standalone test enabled (static build)")

    # Dynamic linking version (commented out - for reference only)
    # # Use dynamic linking like sample_vdec Makefile, with RPATH for device
    # target_link_libraries(test_vdec_standalone PRIVATE
    #     -Wl,--start-group
    #     -lvdec
    #     -lmisc
    #     -lsys
    #     -lvi
    #     -lvpss
    #     -lvo
    #     -lgdc
    #     -lvenc
    #     -lcvi_bin
    #     -lcvi_bin_isp
    #     -lisp
    #     -lisp_algo
    #     -lsns_full
    #     -lae
    #     -laf
    #     -lawb
    #     -Wl,--end-group
    #     -lini
    #     z
    #     m
    #     pthread
    #     dl
    #     atomic
    # )
    # # Set RPATH so dynamic libraries can be found on device
    # set_target_properties(test_vdec_standalone PROPERTIES
    #     BUILD_RPATH "/mnt/system/usr/lib:/mnt/system/usr/lib/3rd"
    #     INSTALL_RPATH "/mnt/system/usr/lib:/mnt/system/usr/lib/3rd"
    # )
endif()
