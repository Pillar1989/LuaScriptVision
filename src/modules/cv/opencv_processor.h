#pragma once

#include "cv_processor.h"

namespace lua_cv {

/**
 * OpenCvProcessor - CPU-based image processing using OpenCV
 *
 * This processor implements all operations using OpenCV's CPU functions.
 * It's used as a fallback when:
 * 1. Input frame is cv::Mat (not hardware VIDEO_FRAME_INFO_S)
 * 2. CVI MPI is not available
 *
 * Performance characteristics (1920x1080 → 640x640):
 * - resize: ~25ms (CPU)
 * - cvtColor: ~5ms (CPU)
 * - crop: ~1ms (zero-copy view, or copy if needed)
 *
 * While slower than VPSS hardware (~8ms), it's still faster than
 * VPSS + memcpy overhead (~65ms) when input is already cv::Mat.
 */
class OpenCvProcessor : public CvProcessor {
public:
    OpenCvProcessor() = default;
    ~OpenCvProcessor() override = default;

    /**
     * Resize using cv::resize
     * Uses INTER_LINEAR interpolation by default
     */
    void resize(Frame& frame, int width, int height) override;

    /**
     * Color space conversion using cv::cvtColor
     */
    void cvtColor(Frame& frame, ColorConversion code) override;

    /**
     * Crop using cv::Mat::operator()(Rect)
     * Zero-copy if possible, otherwise copies
     */
    void crop(Frame& frame, int x, int y, int w, int h) override;
};

} // namespace lua_cv
