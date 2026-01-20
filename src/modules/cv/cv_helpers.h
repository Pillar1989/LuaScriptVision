#pragma once

#include <vector>

#include "cvi_frame.h"
#include "cv_types.h"

namespace tensor {
class Tensor;
}

namespace lua_cv::cv_helpers {

void resize(Frame& frame, int width, int height);
void cvt_color(Frame& frame, ColorConversion code);
void crop(Frame& frame, int x, int y, int w, int h);

const char* get_backend_name(const Frame& frame);

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
