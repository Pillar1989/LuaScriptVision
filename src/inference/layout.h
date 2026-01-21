#pragma once

#include <cstdint>
#include <vector>

namespace inference {

enum class Layout {
    NCHW,
    NHWC,
    NC,
    Unknown
};

inline const char* layout_name(Layout layout) {
    switch (layout) {
        case Layout::NCHW:
            return "nchw";
        case Layout::NHWC:
            return "nhwc";
        case Layout::NC:
            return "nc";
        default:
            return "unknown";
    }
}

inline Layout infer_layout(const std::vector<int64_t>& shape) {
    if (shape.size() == 2) {
        return Layout::NC;
    }
    if (shape.size() != 4) {
        return Layout::Unknown;
    }
    int64_t c_first = shape[1];
    int64_t c_last = shape[3];
    if (c_first > 0 && c_first <= 4) {
        return Layout::NCHW;
    }
    if (c_last > 0 && c_last <= 4) {
        return Layout::NHWC;
    }
    return Layout::Unknown;
}

} // namespace inference
