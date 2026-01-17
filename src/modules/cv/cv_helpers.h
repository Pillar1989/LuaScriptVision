#pragma once

#include "cvi_frame.h"
#include "cv_types.h"

// Forward declaration for tensor::Tensor
namespace tensor {
    class Tensor;
}

namespace lua_cv {

/**
 * Smart CV Helper Functions
 *
 * These functions automatically select the optimal backend (VPSS hardware or OpenCV CPU)
 * based on the input frame type:
 * - If frame.has_physical_addr() == true  → Use VPSS hardware (~8ms for 1080p resize)
 * - If frame.has_physical_addr() == false → Use OpenCV CPU (~25ms for 1080p resize)
 *
 * This provides a unified API for Lua bindings while transparently optimizing performance.
 */
namespace cv_helpers {

/**
 * Smart resize - auto-selects backend
 *
 * @param frame Input/output frame (modified in-place)
 * @param width Target width
 * @param height Target height
 *
 * Performance:
 * - VPSS path: ~8ms for 1920x1080 → 640x640
 * - CPU path:  ~25ms for 1920x1080 → 640x640
 */
void resize(Frame& frame, int width, int height);

/**
 * Smart color conversion - auto-selects backend
 *
 * @param frame Input/output frame (modified in-place)
 * @param code Color conversion code
 *
 * Supported conversions:
 * - VPSS: NV12↔RGB, BGR2RGB (limited)
 * - CPU: All OpenCV color conversions
 */
void cvt_color(Frame& frame, ColorConversion code);

/**
 * Smart crop - auto-selects backend
 *
 * @param frame Input/output frame (modified in-place)
 * @param x Left coordinate
 * @param y Top coordinate
 * @param w Width
 * @param h Height
 */
void crop(Frame& frame, int x, int y, int w, int h);

/**
 * Get the preferred processor for a given frame
 *
 * Useful for testing or explicit backend selection.
 * Returns "vpss" or "opencv" based on frame properties.
 */
const char* get_backend_name(const Frame& frame);

/**
 * Convert Frame to Tensor with normalization
 *
 * @param frame Input frame (will call to_mat() internally)
 * @param scale Scale factor applied before normalization
 * @param mean Mean values per channel (vector size must match channels)
 * @param std Standard deviation per channel (vector size must match channels)
 * @return Tensor in NCHW format (batch=1, channels, height, width)
 *
 * Transformation: output[c][h][w] = (input[h][w][c] * scale - mean[c]) / std[c]
 *
 * Performance: ~5ms for 640x640 RGB image
 */
tensor::Tensor frame_to_tensor(
    const Frame& frame,
    double scale,
    const std::vector<double>& mean,
    const std::vector<double>& std);

} // namespace cv_helpers

} // namespace lua_cv
