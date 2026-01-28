#pragma once

#include <vector>

#include "frame.h"
#include "cv_types.h"

namespace tensor {
class Tensor;
}

namespace lua_cv::cv_helpers {

void resize(Frame& frame, int width, int height);
void cvt_color(Frame& frame, ColorConversion code);
void crop(Frame& frame, int x, int y, int w, int h);

const char* get_backend_name(const Frame& frame);

#ifdef USE_CVI_MPI
// Check if a Frame can be passed directly to TPU via physical address (zero-copy).
// Verifies: CVI storage, physical address present, 64-byte alignment,
// matching pixel format and dimensions.
// If reason is non-null and returns false, a diagnostic string is written.
bool can_zero_copy(const Frame& frame,
                   PIXEL_FORMAT_E required_format,
                   uint32_t required_width,
                   uint32_t required_height,
                   std::string* reason = nullptr);
#endif

// Convert Frame to Tensor with normalization
// Output shape: [1, C, H, W]
// scale: applied before mean/std
// mean/std: optional per-channel normalization
// If mean/std are empty, defaults are 0 and 1.
// If size is 1, the value is applied to all channels.
// If size equals channels, use per-channel values.
// Otherwise, throws invalid_argument.
tensor::Tensor frame_to_tensor(const Frame& frame,
                               double scale,
                               const std::vector<double>& mean,
                               const std::vector<double>& stddev);

} // namespace lua_cv::cv_helpers
