# Targets.cmake - Target creation helper functions
#
# This module provides functions to simplify target creation:
# - target_link_cvi_tpu(): Link TPU static libraries
# - target_link_cvi_mpi(): Link MPI base libraries
# - target_link_cvi_isp(): Link ISP/Camera libraries
# - target_link_gtest(): Link Google Test
# - add_cvi_test(): Create a test executable with common configuration

# =============================================================================
# Link TPU Libraries
# =============================================================================
# Usage: target_link_cvi_tpu(target_name)
function(target_link_cvi_tpu TARGET)
    if(NOT SG200X_BUILD)
        return()
    endif()

    target_include_directories(${TARGET} PRIVATE ${TPU_INCLUDE_DIR})
    target_link_directories(${TARGET} PRIVATE ${TPU_LIB_DIR})
    target_link_libraries(${TARGET} PRIVATE ${CVI_TPU_STATIC_LIBS})
endfunction()

# =============================================================================
# Link MPI Base Libraries (with link group for circular dependencies)
# =============================================================================
# Usage: target_link_cvi_mpi(target_name)
function(target_link_cvi_mpi TARGET)
    if(NOT SG200X_BUILD)
        return()
    endif()

    target_include_directories(${TARGET} PRIVATE ${CVI_MPI_INCLUDE_DIR})
    target_link_directories(${TARGET} PRIVATE ${CVI_MPI_LIB_DIR})
    target_link_libraries(${TARGET} PRIVATE
        -Wl,--start-group
        ${CVI_MPI_BASE_LIBS}
        -Wl,--end-group
    )
    target_compile_definitions(${TARGET} PRIVATE USE_CVI_MPI)
endfunction()

# =============================================================================
# Link ISP/Camera Libraries (requires MPI to be linked first)
# =============================================================================
# Usage: target_link_cvi_isp(target_name)
function(target_link_cvi_isp TARGET)
    if(NOT SG200X_BUILD)
        return()
    endif()

    target_include_directories(${TARGET} PRIVATE ${CVI_ISP_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE
        -Wl,--start-group
        ${CVI_ISP_LIBS}
        ${CVI_MPI_LIB_DIR}/libsys.a
        -Wl,--end-group
    )
    target_compile_definitions(${TARGET} PRIVATE USE_CVI_CAMERA)
endfunction()

# =============================================================================
# Link RTSP Libraries
# =============================================================================
# Usage: target_link_cvi_rtsp(target_name)
function(target_link_cvi_rtsp TARGET)
    if(NOT SG200X_BUILD)
        return()
    endif()

    target_include_directories(${TARGET} PRIVATE ${CVI_RTSP_INCLUDE_DIR})
    target_link_directories(${TARGET} PRIVATE ${CVI_RTSP_LIB_DIR})
    # RTSP depends on live555 (shared only) from sysroot
    target_link_directories(${TARGET} PRIVATE ${ALSA_LIB_DIR})
    target_link_libraries(${TARGET} PRIVATE
        "-Wl,-Bdynamic"
        cvi_rtsp
        liveMedia
        groupsock
        UsageEnvironment
        BasicUsageEnvironment
    )

    # Set RPATH for dynamic library on device
    set_target_properties(${TARGET} PROPERTIES
        BUILD_RPATH "/mnt/system/usr/lib"
        INSTALL_RPATH "/mnt/system/usr/lib"
    )
endfunction()

# =============================================================================
# Link ALSA Libraries
# =============================================================================
# Usage: target_link_alsa(target_name)
function(target_link_alsa TARGET)
    if(NOT SG200X_BUILD)
        return()
    endif()

    target_include_directories(${TARGET} PRIVATE ${ALSA_INCLUDE_DIR})
    target_link_directories(${TARGET} PRIVATE ${ALSA_LIB_DIR})
    # Dynamic linking for ALSA (no static library available)
    target_link_libraries(${TARGET} PRIVATE "-Wl,-Bdynamic" asound "-Wl,-Bstatic")
endfunction()

# =============================================================================
# Link System Libraries
# =============================================================================
# Usage: target_link_system_libs(target_name)
function(target_link_system_libs TARGET)
    target_link_libraries(${TARGET} PRIVATE ${CVI_SYSTEM_LIBS})

    if(UNIX AND NOT APPLE)
        target_link_libraries(${TARGET} PRIVATE dl)
    endif()
endfunction()

# =============================================================================
# Link Google Test
# =============================================================================
# Usage: target_link_gtest(target_name [WITH_MAIN])
function(target_link_gtest TARGET)
    cmake_parse_arguments(ARG "WITH_MAIN" "" "" ${ARGN})

    if(ARG_WITH_MAIN)
        target_link_libraries(${TARGET} PRIVATE
            -Wl,--whole-archive
            gtest_main
            gtest
            -Wl,--no-whole-archive
        )
    else()
        target_link_libraries(${TARGET} PRIVATE
            -Wl,--whole-archive
            gtest
            -Wl,--no-whole-archive
        )
    endif()

    target_link_libraries(${TARGET} PRIVATE pthread)
endfunction()

# =============================================================================
# Create SG200X Test Executable
# =============================================================================
# Usage: add_cvi_test(
#     NAME test_name
#     SOURCES source1.cpp source2.cpp ...
#     [INCLUDES dir1 dir2 ...]
#     [LIBS lib1 lib2 ...]
#     [TPU]          - Link TPU libraries
#     [MPI]          - Link MPI libraries
#     [ISP]          - Link ISP libraries
#     [RTSP]         - Link RTSP libraries
#     [ALSA]         - Link ALSA libraries
#     [OPENCV]       - Link OpenCV
#     [LUA]          - Link Lua
#     [GTEST_MAIN]   - Use gtest_main
#     [STATIC]       - Static linking
#     [DEFINITIONS def1 def2 ...]
# )
function(add_cvi_test)
    cmake_parse_arguments(ARG
        "TPU;MPI;ISP;RTSP;ALSA;OPENCV;LUA;GTEST_MAIN;STATIC"
        "NAME"
        "SOURCES;INCLUDES;LIBS;DEFINITIONS"
        ${ARGN}
    )

    if(NOT ARG_NAME)
        message(FATAL_ERROR "add_cvi_test: NAME is required")
    endif()

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_cvi_test: SOURCES is required")
    endif()

    # Create executable
    add_executable(${ARG_NAME} ${ARG_SOURCES})

    # Static linking
    if(ARG_STATIC AND SG200X_BUILD)
        target_link_libraries(${ARG_NAME} PRIVATE -static)
    endif()

    # Custom includes
    if(ARG_INCLUDES)
        target_include_directories(${ARG_NAME} PRIVATE ${ARG_INCLUDES})
    endif()

    # TPU libraries
    if(ARG_TPU)
        target_link_cvi_tpu(${ARG_NAME})
    endif()

    # Lua (must be linked with whole-archive for static builds)
    if(ARG_LUA)
        target_link_libraries(${ARG_NAME} PRIVATE
            -Wl,--whole-archive
            lua
            -Wl,--no-whole-archive
        )
    endif()

    # Google Test
    if(ARG_GTEST_MAIN)
        target_link_gtest(${ARG_NAME} WITH_MAIN)
    else()
        target_link_gtest(${ARG_NAME})
    endif()

    # MPI libraries
    if(ARG_MPI)
        target_link_cvi_mpi(${ARG_NAME})
    endif()

    # ISP libraries
    if(ARG_ISP)
        target_link_cvi_isp(${ARG_NAME})
    endif()

    # RTSP libraries
    if(ARG_RTSP)
        target_link_cvi_rtsp(${ARG_NAME})
    endif()

    # ALSA libraries
    if(ARG_ALSA)
        target_link_alsa(${ARG_NAME})
    endif()

    # OpenCV
    if(ARG_OPENCV)
        target_include_directories(${ARG_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${ARG_NAME} PRIVATE ${OpenCV_LIBS})
    endif()

    # Custom libraries
    if(ARG_LIBS)
        target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LIBS})
    endif()

    # System libraries
    target_link_system_libs(${ARG_NAME})

    # Compile definitions
    if(ARG_DEFINITIONS)
        target_compile_definitions(${ARG_NAME} PRIVATE ${ARG_DEFINITIONS})
    endif()

    message(STATUS "${ARG_NAME}: Test enabled")
endfunction()

# =============================================================================
# Common Include Directories for src/
# =============================================================================
set(SRC_COMMON_INCLUDES
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/memory"
    "${CMAKE_SOURCE_DIR}/src/modules"
    "${CMAKE_SOURCE_DIR}/src/modules/tensor"
    "${CMAKE_SOURCE_DIR}/src/modules/cv"
    "${CMAKE_SOURCE_DIR}/src/stream"
    "${CMAKE_SOURCE_DIR}/src/pipeline"
    "${CMAKE_SOURCE_DIR}/src/bindings"
    "${CMAKE_SOURCE_DIR}/src/utils"
    "${CMAKE_SOURCE_DIR}/src/inference"
    "${CMAKE_SOURCE_DIR}/src/node"
    "${CMAKE_SOURCE_DIR}/third_party/mongoose"
)

set(LUAINTF_INCLUDES
    "${CMAKE_SOURCE_DIR}/lua-intf-ex/src/include"
    "${CMAKE_SOURCE_DIR}/lua"
)
