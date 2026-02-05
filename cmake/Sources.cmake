# Sources.cmake - Source file collection
#
# This module collects source files into reusable variables

# =============================================================================
# Lua Library (built as C++ for exception safety)
# =============================================================================
file(GLOB LUA_SRC "${CMAKE_SOURCE_DIR}/lua/*.c")
list(REMOVE_ITEM LUA_SRC
    "${CMAKE_SOURCE_DIR}/lua/lua.c"
    "${CMAKE_SOURCE_DIR}/lua/luac.c"
    "${CMAKE_SOURCE_DIR}/lua/onelua.c"
)

add_library(lua STATIC ${LUA_SRC})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(lua PRIVATE -x c++ -O3 -Wall -DLUA_USE_POSIX -flto)
else()
    target_compile_options(lua PRIVATE -x c++ -O3 -Wall -DLUA_USE_POSIX)
endif()
target_include_directories(lua PUBLIC "${CMAKE_SOURCE_DIR}/lua")

# =============================================================================
# LuaIntf (header-only)
# =============================================================================
add_library(LuaIntf INTERFACE)
target_include_directories(LuaIntf INTERFACE "${CMAKE_SOURCE_DIR}/lua-intf-ex/src/include")
target_link_libraries(LuaIntf INTERFACE lua)

# =============================================================================
# Module Source Files
# =============================================================================
file(GLOB MODULE_SOURCES "${CMAKE_SOURCE_DIR}/src/modules/*.cpp")
file(GLOB TENSOR_SOURCES "${CMAKE_SOURCE_DIR}/src/modules/tensor/*.cpp")
file(GLOB MEMORY_SOURCES "${CMAKE_SOURCE_DIR}/src/memory/*.cpp")
file(GLOB CV_SOURCES "${CMAKE_SOURCE_DIR}/src/modules/cv/*.cpp")
file(GLOB BINDING_SOURCES "${CMAKE_SOURCE_DIR}/src/bindings/*.cpp")
file(GLOB INFERENCE_SOURCES "${CMAKE_SOURCE_DIR}/src/inference/*.cpp")
file(GLOB STREAM_SOURCES "${CMAKE_SOURCE_DIR}/src/stream/*.cpp")
file(GLOB PIPELINE_SOURCES "${CMAKE_SOURCE_DIR}/src/pipeline/*.cpp")
file(GLOB NODE_SOURCES "${CMAKE_SOURCE_DIR}/src/node/*.cpp")

# =============================================================================
# T-Head RISC-V Optimized Sources
# =============================================================================
if(RV_THEAD_ENABLED)
    set(TENSOR_RV_THEAD_SOURCES
        ${CMAKE_SOURCE_DIR}/src/modules/tensor/rv-thead/weighted_sum.cpp
        ${CMAKE_SOURCE_DIR}/src/modules/tensor/rv-thead/sigmoid_max.cpp
        ${CMAKE_SOURCE_DIR}/src/modules/tensor/rv-thead/max_with_argmax.cpp
    )
    list(APPEND TENSOR_SOURCES ${TENSOR_RV_THEAD_SOURCES})
    message(STATUS "T-Head tensor optimizations: ${TENSOR_RV_THEAD_SOURCES}")
endif()

# =============================================================================
# RVV Assembly Sources (SG200X only)
# =============================================================================
if(SG200X_BUILD)
    set(INFERENCE_ASM_SOURCES
        "${CMAKE_SOURCE_DIR}/src/inference/shl_c906_u8_to_f32.S"
        "${CMAKE_SOURCE_DIR}/src/inference/shl_c906_i8_to_f32.S"
    )
    list(APPEND INFERENCE_SOURCES ${INFERENCE_ASM_SOURCES})
    message(STATUS "RVV optimized dequantization enabled")
endif()

# =============================================================================
# Conditional Source Exclusion
# =============================================================================
if(USE_ONNX_RUNTIME)
    # CPU-only build: exclude hardware-accelerated components
    list(REMOVE_ITEM CV_SOURCES
        "${CMAKE_SOURCE_DIR}/src/modules/cv/cvi_camera.cpp"
        "${CMAKE_SOURCE_DIR}/src/modules/cv/cvi_sensor.cpp"
        "${CMAKE_SOURCE_DIR}/src/modules/cv/cvi_vpss_processor.cpp"
    )
    list(REMOVE_ITEM NODE_SOURCES
        "${CMAKE_SOURCE_DIR}/src/node/stream_node.cpp"
    )
    message(STATUS "CV module: CPU-only (OpenCV)")
else()
    # SG200X build: conditionally exclude camera sources
    if(NOT ENABLE_CVI_CAMERA)
        list(REMOVE_ITEM CV_SOURCES
            "${CMAKE_SOURCE_DIR}/src/modules/cv/cvi_camera.cpp"
            "${CMAKE_SOURCE_DIR}/src/modules/cv/cvi_sensor.cpp"
        )
        message(STATUS "CV module: Hardware-accelerated (VPSS only, Camera disabled)")
    else()
        message(STATUS "CV module: Hardware-accelerated (Camera + ISP + VPSS enabled)")
    endif()

    add_compile_definitions(USE_CVI_MPI)
endif()

# =============================================================================
# Combined Source Sets
# =============================================================================
set(ALL_MODULE_SOURCES
    ${MODULE_SOURCES}
    ${TENSOR_SOURCES}
    ${MEMORY_SOURCES}
    ${CV_SOURCES}
    ${BINDING_SOURCES}
    ${INFERENCE_SOURCES}
    ${STREAM_SOURCES}
    ${PIPELINE_SOURCES}
    ${NODE_SOURCES}
)
