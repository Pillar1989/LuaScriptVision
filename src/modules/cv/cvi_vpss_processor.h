#pragma once

#include "cv_processor.h"

#ifdef USE_CVI_MPI
#include <cvi_vpss.h>
#include <cvi_vb.h>
#include <cvi_buffer.h>
#include "mmf_context.h"
#endif

namespace lua_cv {

class CviVpssProcessor : public CvProcessor {
public:
    struct LetterboxMeta {
        float scale = 0.0f;
        int pad_x = 0;
        int pad_y = 0;
        int ori_w = 0;
        int ori_h = 0;
    };

    CviVpssProcessor();
    ~CviVpssProcessor() override;

    void resize(Frame& frame, int width, int height) override;
    void cvtColor(Frame& frame, ColorConversion code) override;
    void crop(Frame& frame, int x, int y, int w, int h) override;
    void convert_format(Frame& frame, PixelFormat output_format);
    void letterbox(Frame& frame, int width, int height, uint8_t pad_value,
                   LetterboxMeta* meta = nullptr,
                   PixelFormat output_format = PixelFormat::RGB_PLANAR);

    // Combined crop and resize in single VPSS pass (for ROI inference)
    void crop_resize(Frame& frame,
                     int crop_x, int crop_y, int crop_w, int crop_h,
                     int out_width, int out_height,
                     PixelFormat output_format = PixelFormat::RGB);

    int last_error() const;

#ifdef USE_CVI_MPI
    VIDEO_FRAME_INFO_S mat_to_video_frame(const cv::Mat& mat, VB_BLK& vb_block);
    VIDEO_FRAME_INFO_S mat_to_video_frame(const cv::Mat& mat, VB_BLK& vb_block, VB_POOL pool);
    void destroy_group();  // Public method for explicit cleanup before MMF shutdown
#endif

private:
#ifdef USE_CVI_MPI
    void ensure_group(uint32_t input_width, uint32_t input_height, PixelFormat input_format);
    void ensure_channel(uint32_t out_width, uint32_t out_height, PixelFormat out_format,
                        bool letterbox, uint8_t pad_value);
    void process_frame(Frame& frame, uint32_t out_width, uint32_t out_height, PixelFormat out_format,
                       bool use_crop, int crop_x, int crop_y, int crop_w, int crop_h,
                       bool letterbox, uint8_t pad_value);
    VB_POOL select_output_pool(uint32_t width, uint32_t height, PixelFormat format) const;

    VPSS_GRP grp_ = -1;
    VPSS_CHN chn_ = 0;
    bool group_created_ = false;
    bool group_started_ = false;

    uint32_t max_width_ = 0;
    uint32_t max_height_ = 0;
    PixelFormat input_format_ = PixelFormat::UNKNOWN;
    uint32_t out_width_ = 0;
    uint32_t out_height_ = 0;
    PixelFormat out_format_ = PixelFormat::UNKNOWN;
    VB_POOL out_pool_ = VB_INVALID_POOLID;
    bool letterbox_enabled_ = false;
    uint8_t pad_value_ = 0;
    CVI_S32 last_error_ = CVI_SUCCESS;
#endif
};

} // namespace lua_cv
