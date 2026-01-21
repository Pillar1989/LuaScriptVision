#pragma once

#include <cstdint>

#include "cvi_frame.h"
#include "inference/tensor_descriptor.h"
#include "tensor/tensor.h"

namespace lua_cv {

tensor::Tensor frame_to_tensor(const Frame& frame,
                               const inference::TensorDescriptor& desc);

Frame tensor_to_frame(const tensor::Tensor& tensor);

uint64_t get_physical_address(const Frame& frame);

} // namespace lua_cv
