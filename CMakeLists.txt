# =============================================================================
# LuaScriptVision - CMakeLists.txt
# =============================================================================
# Hybrid C++/Lua inference framework for embedded Linux systems
#
# Build modes:
#   - CPU-only (default): cmake -B build
#   - SG200X TPU: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake -B build
#
# Options:
#   - ENABLE_CVI_CAMERA: Enable camera support with ISP (default: OFF)
#   - USE_VDEC_DECODE: Use hardware VDEC for JPEG decode (default: OFF)
#   - ENABLE_RV_THEAD: Enable T-Head RISC-V optimizations (default: ON)

cmake_minimum_required(VERSION 3.18)
project(ModelInfer LANGUAGES C CXX ASM)

# =============================================================================
# 1. Platform Configuration
# =============================================================================
include(cmake/Platform.cmake)

# =============================================================================
# 2. External Dependencies
# =============================================================================
include(cmake/Dependencies.cmake)

# =============================================================================
# 3. Library Definitions (SG200X only)
# =============================================================================
include(cmake/Libraries.cmake)

# =============================================================================
# 4. Target Helper Functions
# =============================================================================
include(cmake/Targets.cmake)

# =============================================================================
# 5. Source Files
# =============================================================================
include(cmake/Sources.cmake)

# =============================================================================
# 6. Main Executable: lua_runner
# =============================================================================
add_executable(lua_runner
    examples/lua_runner/main.cpp
    ${ALL_MODULE_SOURCES}
)

target_include_directories(lua_runner PRIVATE
    ${SRC_COMMON_INCLUDES}
    ${NLOHMANN_JSON_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(lua_runner PRIVATE LuaIntf ${OpenCV_LIBS})

if(USE_ONNX_RUNTIME)
    # CPU-only build
    target_link_libraries(lua_runner PRIVATE lua onnxruntime)
    target_include_directories(lua_runner PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    set_target_properties(lua_runner PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
        INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    )
else()
    # SG200X build
    target_include_directories(lua_runner PRIVATE
        ${CVI_BASE_INCLUDE_DIRS}
        ${ALSA_INCLUDE_DIR}
    )

    target_link_directories(lua_runner PRIVATE
        ${CVI_BASE_LINK_DIRS}
        ${ALSA_LIB_DIR}
        ${MOSQUITTO_LIB_DIR}
    )

    target_link_alsa(lua_runner)
    target_link_libraries(lua_runner PRIVATE
        "-Wl,-Bdynamic"
        mosquitto
        ssl
        crypto
        cares
    )
    target_link_cvi_rtsp(lua_runner)

    if(ENABLE_CVI_CAMERA)
        target_include_directories(lua_runner PRIVATE ${CVI_ISP_INCLUDE_DIR})

        target_link_libraries(lua_runner PRIVATE
            -static
            -Wl,--whole-archive lua -Wl,--no-whole-archive
            ${CVI_TPU_STATIC_LIBS}
            -Wl,--start-group
            ${CVI_ISP_LIBS}
            ${CVI_MPI_BASE_LIBS}
            -Wl,--end-group
        )
    else()
        target_link_libraries(lua_runner PRIVATE
            -static
            -Wl,--whole-archive lua -Wl,--no-whole-archive
            ${CVI_TPU_STATIC_LIBS}
            ${CVI_MPI_BASE_LIBS}
        )
    endif()

    target_link_system_libs(lua_runner)
    message(STATUS "lua_runner: Sophgo TPU support linked")
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(lua_runner PRIVATE dl)
endif()

# =============================================================================
# 7. Pure C++ Inference App: cpp_infer
# =============================================================================
add_executable(cpp_infer
    examples/cpp_runner/main.cpp
    ${INFERENCE_SOURCES}
)

target_include_directories(cpp_infer PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cpp_infer PRIVATE ${OpenCV_LIBS})

if(USE_ONNX_RUNTIME)
    target_link_libraries(cpp_infer PRIVATE onnxruntime)
    target_include_directories(cpp_infer PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    set_target_properties(cpp_infer PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
        INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    )
else()
    target_sources(cpp_infer PRIVATE
        src/memory/tpu_memory.cpp
        src/memory/device_buffer.cpp
        src/memory/cpu_memory.cpp
        src/memory/sync_handle.cpp
    )
    target_include_directories(cpp_infer PRIVATE
        ${TPU_INCLUDE_DIR}
        ${CVI_MPI_INCLUDE_DIR}
        "${CMAKE_SOURCE_DIR}/src/memory"
    )
    target_link_directories(cpp_infer PRIVATE ${TPU_LIB_DIR})
    target_link_libraries(cpp_infer PRIVATE
        -static
        ${CVI_TPU_STATIC_LIBS}
    )
    target_link_system_libs(cpp_infer)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(cpp_infer PRIVATE dl)
endif()

message(STATUS "cpp_infer: inference benchmark enabled")

# =============================================================================
# 8. Parallel Inference + Streaming Application (SG200X with Camera)
# =============================================================================
if(SG200X_BUILD AND ENABLE_CVI_CAMERA)
    add_executable(parallel_infer_stream
        examples/parallel_infer_stream/main.cpp
        ${ALL_MODULE_SOURCES}
    )

    target_include_directories(parallel_infer_stream PRIVATE
        ${SRC_COMMON_INCLUDES}
        ${NLOHMANN_JSON_INCLUDE}
        ${OpenCV_INCLUDE_DIRS}
        ${CVI_BASE_INCLUDE_DIRS}
        ${CVI_ISP_INCLUDE_DIR}
        ${CVI_RTSP_INCLUDE_DIR}
        ${ALSA_INCLUDE_DIR}
        ${MOSQUITTO_INCLUDE_DIR}
    )

    target_link_directories(parallel_infer_stream PRIVATE
        ${CVI_BASE_LINK_DIRS}
        ${CVI_RTSP_LIB_DIR}
        ${ALSA_LIB_DIR}
        ${MOSQUITTO_LIB_DIR}
    )

    target_link_libraries(parallel_infer_stream PRIVATE
        LuaIntf
        ${OpenCV_LIBS}
        ${CVI_TPU_STATIC_LIBS}
        -Wl,--whole-archive lua -Wl,--no-whole-archive
        -Wl,--start-group
        ${CVI_ISP_LIBS}
        ${CVI_MPI_BASE_LIBS}
        -Wl,--end-group
        cvi_rtsp
        asound
        mosquitto
        ssl
        crypto
        cares
    )

    target_link_system_libs(parallel_infer_stream)

    set_target_properties(parallel_infer_stream PROPERTIES
        BUILD_RPATH "/mnt/system/usr/lib"
        INSTALL_RPATH "/mnt/system/usr/lib"
    )

    target_compile_definitions(parallel_infer_stream PRIVATE
        USE_CVI_MPI USE_CVI_CAMERA USE_CVI_TPU
    )

    message(STATUS "parallel_infer_stream: Parallel inference + streaming application enabled")
endif()

# =============================================================================
# 9. Node Server (SG200X only)
# =============================================================================
if(SG200X_BUILD)
    add_executable(node_server
        examples/node_server/main.cpp
        ${ALL_MODULE_SOURCES}
    )

    target_include_directories(node_server PRIVATE
        ${SRC_COMMON_INCLUDES}
        ${NLOHMANN_JSON_INCLUDE}
        ${MOSQUITTO_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CVI_BASE_INCLUDE_DIRS}
    )

    target_link_directories(node_server PRIVATE
        ${MOSQUITTO_LIB_DIR}
        ${CVI_BASE_LINK_DIRS}
    )

    target_link_libraries(node_server PRIVATE
        LuaIntf
        ${OpenCV_LIBS}
        mosquitto
        ssl
        crypto
        cares
    )
    target_link_libraries(node_server PRIVATE
        -Wl,--whole-archive
        lua
        -Wl,--no-whole-archive
    )

    target_link_cvi_tpu(node_server)
    target_link_cvi_mpi(node_server)
    target_link_cvi_rtsp(node_server)
    if(ENABLE_CVI_CAMERA)
        target_include_directories(node_server PRIVATE ${CVI_ISP_INCLUDE_DIR})
        target_link_cvi_isp(node_server)
    endif()
    target_link_system_libs(node_server)

    set_target_properties(node_server PROPERTIES
        BUILD_RPATH "${MOSQUITTO_LIB_DIR};/mnt/system/usr/lib"
        INSTALL_RPATH "/usr/lib"
    )

    message(STATUS "node_server: MQTT node server enabled")
endif()

# =============================================================================
# 10. Test Programs (SG200X only)
# =============================================================================
if(SG200X_BUILD)
    include(cmake/Tests.cmake)
    include(cmake/NodeTests.cmake)
endif()
