#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

namespace inference {

struct QuantParams {
    float scale = 1.0f;
    int32_t zero_point = 0;

    bool is_identity() const {
        return scale == 1.0f && zero_point == 0;
    }

    float dequantize(int32_t value) const {
        if (scale == 0.0f) {
            return 0.0f;
        }
        return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
    }

    int32_t quantize(float value) const {
        if (scale == 0.0f) {
            return zero_point;
        }
        double scaled = static_cast<double>(value) / static_cast<double>(scale) +
                        static_cast<double>(zero_point);
        long long q = std::llround(scaled);
        long long min_val = static_cast<long long>(std::numeric_limits<int32_t>::min());
        long long max_val = static_cast<long long>(std::numeric_limits<int32_t>::max());
        if (q < min_val) {
            q = min_val;
        } else if (q > max_val) {
            q = max_val;
        }
        return static_cast<int32_t>(q);
    }
};

} // namespace inference
