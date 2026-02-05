# Tests.cmake - Test target definitions
#
# This module defines all test executables for the SG200X platform

if(NOT SG200X_BUILD)
    message(STATUS "Tests: Skipped (not SG200X build)")
    return()
endif()

# =============================================================================
# Google Test Setup
# =============================================================================
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest third_party/googletest)
else()
    message(FATAL_ERROR "googletest not found at third_party/googletest")
endif()

# =============================================================================
# Common Test Includes
# =============================================================================
set(TEST_COMMON_INCLUDES
    "${CMAKE_SOURCE_DIR}/tests"
    ${SRC_COMMON_INCLUDES}
)

if(ENABLE_CVI_CAMERA)
    list(APPEND TEST_COMMON_INCLUDES ${CVI_ISP_INCLUDE_DIR})
endif()

# =============================================================================
# Test 1: TPU Memory Management
# =============================================================================
add_cvi_test(
    NAME test_tpu_memory
    SOURCES
        tests/test_tpu_memory.cpp
        src/memory/device_buffer.cpp
        src/memory/cpu_memory.cpp
        src/memory/tpu_memory.cpp
        src/memory/sync_handle.cpp
    INCLUDES
        ${TEST_COMMON_INCLUDES}
    TPU
    MPI
    GTEST_MAIN
    STATIC
)

# =============================================================================
# Test 1.1: VB Pool Manager
# =============================================================================
add_cvi_test(
    NAME test_vb_pool
    SOURCES
        tests/test_vb_pool.cpp
        src/memory/vb_pool_manager.cpp
        src/memory/vb_memory.cpp
        src/memory/device_buffer.cpp
        src/memory/cpu_memory.cpp
        src/memory/sync_handle.cpp
        src/memory/tpu_memory.cpp
    INCLUDES
        ${TEST_COMMON_INCLUDES}
    TPU
    MPI
    GTEST_MAIN
    STATIC
)

# =============================================================================
# Test 2: CV Module Comprehensive Test
# =============================================================================
set(TEST_CV_FUNC_SOURCES
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
    tests/test_input_source_image.cpp
    src/modules/cv/cvi_vpss_processor.cpp
    src/modules/cv/ion_image_loader.cpp
    src/modules/cv/hw_jpeg_decoder.cpp
    src/modules/cv/opencv_processor.cpp
    src/modules/cv/frame.cpp
    src/modules/cv/cv_helpers.cpp
    src/modules/cv/cv_types.cpp
    src/modules/cv/image_source.cpp
    src/modules/cv/camera_source.cpp
    src/modules/cv/mmf_context.cpp
    ${TENSOR_SOURCES}
    ${MEMORY_SOURCES}
)

add_executable(test_cv_func ${TEST_CV_FUNC_SOURCES})

target_include_directories(test_cv_func PRIVATE
    ${TEST_COMMON_INCLUDES}
    ${LUAINTF_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_cv_func PRIVATE -static lua)
target_link_cvi_tpu(test_cv_func)
target_link_gtest(test_cv_func)
target_link_cvi_mpi(test_cv_func)
target_link_libraries(test_cv_func PRIVATE ${OpenCV_LIBS})
target_link_system_libs(test_cv_func)

if(ENABLE_CVI_CAMERA)
    target_sources(test_cv_func PRIVATE
        src/modules/cv/cvi_camera.cpp
        src/modules/cv/cvi_sensor.cpp
    )
    target_link_cvi_isp(test_cv_func)
    message(STATUS "test_cv_func: Camera capture tests enabled")
endif()

message(STATUS "test_cv_func: CV module comprehensive test enabled")

# =============================================================================
# Test 2.1: CviSession VB Input Test
# =============================================================================
add_cvi_test(
    NAME test_cvi_session_vb
    SOURCES
        tests/test_cvi_session_vb.cpp
        tests/test_common.cpp
        src/inference/cvi_session.cpp
        src/inference/shl_c906_u8_to_f32.S
        src/inference/shl_c906_i8_to_f32.S
        src/memory/tpu_memory.cpp
        src/memory/vb_memory.cpp
        src/memory/vb_pool_manager.cpp
        src/memory/device_buffer.cpp
        src/memory/cpu_memory.cpp
        src/memory/sync_handle.cpp
        src/modules/cv/mmf_context.cpp
        src/modules/cv/cv_types.cpp
    INCLUDES
        ${TEST_COMMON_INCLUDES}
        ${OpenCV_INCLUDE_DIRS}
    TPU
    MPI
    OPENCV
    STATIC
)

if(ENABLE_CVI_CAMERA)
    target_sources(test_cvi_session_vb PRIVATE
        src/modules/cv/cvi_sensor.cpp
    )
    target_link_cvi_isp(test_cvi_session_vb)
endif()

# =============================================================================
# Test 2.2: VPSS Capability Validation (local, no device required)
# =============================================================================
add_executable(test_vpss_capability
    tests/test_vpss_capability.cpp
    tests/vpss_capability.cpp
    src/modules/cv/preprocess_config.cpp
)

target_include_directories(test_vpss_capability PRIVATE
    ${TEST_COMMON_INCLUDES}
    ${CVI_MPI_INCLUDE_DIR}
)

target_link_gtest(test_vpss_capability)
target_link_libraries(test_vpss_capability PRIVATE pthread)

message(STATUS "test_vpss_capability: VPSS capability validation test enabled (local)")

# =============================================================================
# Test 2.3: End-to-End Pipeline Test
# =============================================================================
set(TEST_E2E_SOURCES
    tests/test_end_to_end_pipeline.cpp
    tests/test_common.cpp
    src/inference/cvi_session.cpp
    src/inference/shl_c906_u8_to_f32.S
    src/inference/shl_c906_i8_to_f32.S
    src/memory/tpu_memory.cpp
    src/memory/vb_memory.cpp
    src/memory/vb_pool_manager.cpp
    src/memory/device_buffer.cpp
    src/memory/cpu_memory.cpp
    src/memory/sync_handle.cpp
    src/modules/cv/mmf_context.cpp
    src/modules/cv/cv_types.cpp
    src/modules/cv/image_source.cpp
    src/modules/cv/hw_jpeg_decoder.cpp
    src/modules/cv/frame.cpp
    src/modules/cv/cvi_vpss_processor.cpp
    src/modules/cv/cv_helpers.cpp
)

add_executable(test_end_to_end_pipeline ${TEST_E2E_SOURCES})

target_include_directories(test_end_to_end_pipeline PRIVATE
    ${TEST_COMMON_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_end_to_end_pipeline PRIVATE -static)
target_link_cvi_tpu(test_end_to_end_pipeline)
target_link_gtest(test_end_to_end_pipeline)
target_link_cvi_mpi(test_end_to_end_pipeline)
target_link_libraries(test_end_to_end_pipeline PRIVATE ${OpenCV_LIBS})
target_link_system_libs(test_end_to_end_pipeline)

if(ENABLE_CVI_CAMERA)
    target_sources(test_end_to_end_pipeline PRIVATE
        src/modules/cv/cvi_camera.cpp
        src/modules/cv/cvi_sensor.cpp
        src/modules/cv/camera_source.cpp
    )
    target_link_cvi_isp(test_end_to_end_pipeline)
    message(STATUS "test_end_to_end_pipeline: End-to-end tests with camera enabled")
else()
    message(STATUS "test_end_to_end_pipeline: End-to-end tests with image only")
endif()

# =============================================================================
# Test 3: VDEC Standalone Test
# =============================================================================
add_executable(test_vdec_standalone tests/test_vdec_standalone.cpp)

target_include_directories(test_vdec_standalone PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
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
    ${CVI_TPU_STATIC_LIBS}
)

target_link_gtest(test_vdec_standalone)

target_link_libraries(test_vdec_standalone PRIVATE
    ${CVI_MPI_LIB_DIR}/3rd/libini.a
)

target_link_system_libs(test_vdec_standalone)
target_compile_definitions(test_vdec_standalone PRIVATE USE_CVI_MPI)

message(STATUS "test_vdec_standalone: VDEC standalone test enabled")

# =============================================================================
# Test 4: VENC Encoder Test
# =============================================================================
add_cvi_test(
    NAME test_venc_encoder
    SOURCES
        tests/test_venc_encoder.cpp
        src/stream/venc_encoder.cpp
        src/modules/cv/mmf_context.cpp
        src/modules/cv/cv_types.cpp
    INCLUDES
        ${TEST_COMMON_INCLUDES}
    TPU
    MPI
    STATIC
    DEFINITIONS USE_CVI_MPI
)

# =============================================================================
# Test 5: RTSP Server Test (dynamic linking for libcvi_rtsp.so)
# =============================================================================
add_executable(test_rtsp_server
    tests/test_rtsp_server.cpp
    src/stream/rtsp_server.cpp
    src/stream/venc_encoder.cpp
    src/modules/cv/mmf_context.cpp
    src/modules/cv/cv_types.cpp
)

target_include_directories(test_rtsp_server PRIVATE
    ${TEST_COMMON_INCLUDES}
    ${CVI_RTSP_INCLUDE_DIR}
)

target_link_cvi_tpu(test_rtsp_server)
target_link_gtest(test_rtsp_server)
target_link_cvi_mpi(test_rtsp_server)
target_link_cvi_rtsp(test_rtsp_server)
target_link_system_libs(test_rtsp_server)
target_compile_definitions(test_rtsp_server PRIVATE USE_CVI_MPI)

message(STATUS "test_rtsp_server: RTSP server test enabled (dynamic linking)")

# =============================================================================
# Test 6: Multi-Channel Test (requires camera support)
# =============================================================================
if(ENABLE_CVI_CAMERA)
    add_executable(test_multi_channel
        tests/test_multi_channel.cpp
        src/stream/rtsp_server.cpp
        src/stream/venc_encoder.cpp
        src/modules/cv/mmf_context.cpp
        src/modules/cv/cv_types.cpp
        src/modules/cv/cvi_camera.cpp
        src/modules/cv/cvi_sensor.cpp
        src/modules/cv/frame.cpp
    )

    target_include_directories(test_multi_channel PRIVATE
        ${TEST_COMMON_INCLUDES}
        ${CVI_ISP_INCLUDE_DIR}
        ${CVI_RTSP_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(test_multi_channel PRIVATE ${OpenCV_LIBS})
    target_link_cvi_tpu(test_multi_channel)
    target_link_gtest(test_multi_channel)
    target_link_cvi_isp(test_multi_channel)
    target_link_cvi_mpi(test_multi_channel)
    target_link_cvi_rtsp(test_multi_channel)
    target_link_system_libs(test_multi_channel)
    target_compile_definitions(test_multi_channel PRIVATE USE_CVI_MPI USE_CVI_CAMERA)

    message(STATUS "test_multi_channel: Multi-channel camera test enabled")
endif()

# =============================================================================
# Test 7: Parallel Pipeline Test (requires camera support)
# =============================================================================
if(ENABLE_CVI_CAMERA)
    add_executable(test_parallel_pipeline
        tests/test_parallel_pipeline.cpp
        src/pipeline/parallel_pipeline.cpp
        src/stream/rtsp_server.cpp
        src/stream/venc_encoder.cpp
        src/stream/audio_capture.cpp
        src/modules/cv/mmf_context.cpp
        src/modules/cv/cv_types.cpp
        src/modules/cv/cvi_camera.cpp
        src/modules/cv/cvi_sensor.cpp
        src/modules/cv/frame.cpp
    )

    target_include_directories(test_parallel_pipeline PRIVATE
        ${TEST_COMMON_INCLUDES}
        ${CVI_ISP_INCLUDE_DIR}
        ${CVI_RTSP_INCLUDE_DIR}
        ${ALSA_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(test_parallel_pipeline PRIVATE ${OpenCV_LIBS})
    target_link_cvi_tpu(test_parallel_pipeline)
    target_link_gtest(test_parallel_pipeline)
    target_link_cvi_isp(test_parallel_pipeline)
    target_link_cvi_mpi(test_parallel_pipeline)
    target_link_cvi_rtsp(test_parallel_pipeline)
    target_link_alsa(test_parallel_pipeline)
    target_link_system_libs(test_parallel_pipeline)
    target_compile_definitions(test_parallel_pipeline PRIVATE USE_CVI_MPI USE_CVI_CAMERA)

    message(STATUS "test_parallel_pipeline: Parallel pipeline test enabled")
endif()

message(STATUS "SG200X tests configured")
