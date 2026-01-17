#pragma once

#include "cv_processor.h"

#ifdef USE_CVI_MPI
#include <linux/cvi_comm_video.h>
#include <cvi_vpss.h>
#include <cvi_vb.h>
#endif

namespace lua_cv {

#ifdef USE_CVI_MPI

/**
 * CviVpssProcessor - Hardware-accelerated image processing using SG200X VPSS
 *
 * This processor uses the VPSS (Video Post-Processing Subsystem) hardware
 * to perform image operations with zero-copy when possible.
 *
 * Resource allocation:
 * - VPSS_GRP: 1 (separate from camera pipeline which uses GRP=0)
 * - VPSS_CHN: 0
 * - VB Pool: Dynamically allocated based on frame size
 *
 * Performance characteristics (1920x1080 → 640x640):
 * - resize: ~8ms (hardware)
 * - Zero-copy: No memcpy if input is VIDEO_FRAME_INFO_S
 *
 * Pipeline caching:
 * - VPSS configuration is cached to avoid re-initialization
 * - Only recreates pipeline if input/output dimensions change
 */
class CviVpssProcessor : public CvProcessor {
public:
    CviVpssProcessor();
    ~CviVpssProcessor() override;

    /**
     * Resize using VPSS hardware
     * Supports zero-copy when input has physical address
     */
    void resize(Frame& frame, int width, int height) override;

    /**
     * Color space conversion using VPSS
     * Supports NV21 ↔ RGB888 conversions
     */
    void cvtColor(Frame& frame, ColorConversion code) override;

    /**
     * Crop using VPSS
     */
    void crop(Frame& frame, int x, int y, int w, int h) override;

    /**
     * Convert cv::Mat to VIDEO_FRAME_INFO_S by copying to VB pool
     *
     * This enables hardware acceleration for file inputs.
     * Allocates a VB block and copies Mat data with proper stride alignment.
     *
     * @param mat Input cv::Mat (must be continuous and supported format)
     * @param out_vb_block Output VB block handle (caller must release via CVI_VB_ReleaseBlock)
     * @param vb_pool VB pool ID to allocate from (VB_INVALID_POOLID for public pool)
     * @return VIDEO_FRAME_INFO_S structure pointing to VB pool memory
     * @throws std::runtime_error if allocation or copy fails
     */
    VIDEO_FRAME_INFO_S mat_to_video_frame(const cv::Mat& mat, VB_BLK& out_vb_block, VB_POOL vb_pool = VB_INVALID_POOLID);

private:
    /**
     * Initialize or reconfigure VPSS pipeline
     *
     * @param input_w Input width
     * @param input_h Input height
     * @param input_fmt Input pixel format
     * @param output_w Output width
     * @param output_h Output height
     * @param output_fmt Output pixel format
     * @return CVI_SUCCESS on success, error code otherwise
     */
    CVI_S32 init_vpss_pipeline(
        uint32_t input_w, uint32_t input_h, PIXEL_FORMAT_E input_fmt,
        uint32_t output_w, uint32_t output_h, PIXEL_FORMAT_E output_fmt);

    /**
     * Cleanup VPSS pipeline
     */
    void cleanup_vpss_pipeline();

    /**
     * Process a frame through VPSS
     *
     * @param input Input VIDEO_FRAME_INFO_S
     * @param output Output VIDEO_FRAME_INFO_S (caller must release)
     * @return CVI_SUCCESS on success, error code otherwise
     */
    CVI_S32 vpss_process_frame(
        const VIDEO_FRAME_INFO_S& input,
        VIDEO_FRAME_INFO_S& output);

    // ========== VPSS Resources ==========

    VPSS_GRP vpss_grp_;
    VPSS_CHN vpss_chn_;
    VB_POOL vb_pool_;
    bool initialized_;

    // Pipeline configuration cache (to avoid re-initialization)
    uint32_t cached_input_w_;
    uint32_t cached_input_h_;
    uint32_t cached_output_w_;
    uint32_t cached_output_h_;
    PIXEL_FORMAT_E cached_input_fmt_;
    PIXEL_FORMAT_E cached_output_fmt_;
};

#endif // USE_CVI_MPI

} // namespace lua_cv
