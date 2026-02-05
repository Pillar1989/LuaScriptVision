# Libraries.cmake - Library collection definitions
#
# This module defines reusable library sets for SG200X platform:
# - TPU static libraries
# - CVI MPI base libraries
# - ISP/Camera libraries
# - Common system libraries

if(NOT SG200X_BUILD)
    return()
endif()

# =============================================================================
# TPU Static Libraries
# =============================================================================
set(CVI_TPU_STATIC_LIBS
    ${TPU_LIB_DIR}/libcviruntime-static.a
    ${TPU_LIB_DIR}/libcvikernel-static.a
    ${TPU_LIB_DIR}/libcvimath-static.a
    ${TPU_LIB_DIR}/libflatbuffers.a
)

# =============================================================================
# CVI MPI Base Libraries (sys, vpss, vdec, venc, vo)
# =============================================================================
set(CVI_MPI_BASE_LIBS
    ${CVI_MPI_LIB_DIR}/libsys.a
    ${CVI_MPI_LIB_DIR}/libvpss.a
    ${CVI_MPI_LIB_DIR}/libvdec.a
    ${CVI_MPI_LIB_DIR}/libvenc.a
    ${CVI_MPI_LIB_DIR}/libvo.a
)

# =============================================================================
# ISP/Camera Libraries
# =============================================================================
set(CVI_ISP_LIBS
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

# =============================================================================
# Common System Libraries
# =============================================================================
set(CVI_SYSTEM_LIBS
    z
    m
    pthread
    dl
    atomic
)

# =============================================================================
# Include Directories
# =============================================================================
set(CVI_BASE_INCLUDE_DIRS
    ${TPU_INCLUDE_DIR}
    ${CVI_MPI_INCLUDE_DIR}
)

set(CVI_RTSP_INCLUDE_DIRS
    ${CVI_RTSP_INCLUDE_DIR}
)

set(CVI_ISP_INCLUDE_DIRS
    ${CVI_ISP_INCLUDE_DIR}
)

# =============================================================================
# Link Directories
# =============================================================================
set(CVI_BASE_LINK_DIRS
    ${TPU_LIB_DIR}
    ${CVI_MPI_LIB_DIR}
)

set(CVI_RTSP_LINK_DIRS
    ${CVI_RTSP_LIB_DIR}
)

# =============================================================================
# Combined Library Sets for Common Use Cases
# =============================================================================

# TPU-only (no MPI): For pure inference tests
set(CVI_TPU_ONLY_LIBS
    ${CVI_TPU_STATIC_LIBS}
    ${CVI_SYSTEM_LIBS}
)

# TPU + MPI Base: For VPSS processing without camera
set(CVI_TPU_MPI_LIBS
    ${CVI_TPU_STATIC_LIBS}
    ${CVI_MPI_BASE_LIBS}
    ${CVI_SYSTEM_LIBS}
)

# Full stack: TPU + MPI + ISP (requires ENABLE_CVI_CAMERA)
set(CVI_FULL_LIBS
    ${CVI_TPU_STATIC_LIBS}
    ${CVI_ISP_LIBS}
    ${CVI_MPI_BASE_LIBS}
    ${CVI_SYSTEM_LIBS}
)

message(STATUS "CVI library sets defined")
