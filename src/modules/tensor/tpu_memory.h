#pragma once

#ifdef USE_CVI_TPU

#include "cvi_tpu_memory.h"

namespace tensor {

using TpuMemory = CviTpuMemory;

} // namespace tensor

#endif  // USE_CVI_TPU
