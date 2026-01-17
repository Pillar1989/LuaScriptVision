#pragma once

#include "cvi_frame.h"
#include "cv_types.h"

namespace lua_cv {

/**
 * CvProcessor - Abstract interface for image processing operations
 *
 * This interface provides a common API for image processing that can be
 * implemented by different backends:
 * - VpssProcessor: Hardware-accelerated processing using SG200X VPSS
 * - OpenCvProcessor: CPU-based processing using OpenCV
 *
 * The processor is responsible for common operations like resize, color
 * conversion, cropping, etc. The backend is selected based on the input
 * frame type (has_physical_addr() determines hardware vs CPU path).
 */
class CvProcessor {
public:
    virtual ~CvProcessor() = default;

    /**
     * Resize frame to specified dimensions
     *
     * @param frame Input/output frame (modified in-place or replaced)
     * @param width Target width
     * @param height Target height
     *
     * Behavior:
     * - VpssProcessor: Uses hardware VPSS resize (~8ms for 1080p→640x640)
     * - OpenCvProcessor: Uses cv::resize (~25ms for 1080p→640x640)
     */
    virtual void resize(Frame& frame, int width, int height) = 0;

    /**
     * Convert color space
     *
     * @param frame Input/output frame
     * @param code Color conversion code (e.g., BGR2RGB)
     */
    virtual void cvtColor(Frame& frame, ColorConversion code) = 0;

    /**
     * Crop frame to specified region
     *
     * @param frame Input/output frame
     * @param x Left coordinate
     * @param y Top coordinate
     * @param w Width of crop region
     * @param h Height of crop region
     */
    virtual void crop(Frame& frame, int x, int y, int w, int h) = 0;
};

} // namespace lua_cv
