/**
 * preprocess_op.h - Low-level preprocessing operation sequence
 *
 * Provides fine-grained control over preprocessing pipeline for advanced use cases.
 * Designed with factory methods for Lua-friendly API (Phase 3 binding).
 * Header-only implementation for zero-cost abstraction.
 */

#pragma once

#include <cstdint>
#include <linux/cvi_comm_video.h>

namespace lua_cv {

struct PreprocessOp {
    enum Type {
        CROP,
        RESIZE,
        PAD,
        COLOR,
        ROTATE,
        FLIP
    };

    Type type;

    union {
        struct {
            uint32_t x, y, width, height;
        } crop;

        struct {
            uint32_t width, height;
            bool keep_aspect_ratio;
        } resize;

        struct {
            uint32_t top, bottom, left, right;
            uint8_t value;
        } pad;

        struct {
            PIXEL_FORMAT_E format;
        } color;

        struct {
            int angle_degrees;
        } rotate;

        struct {
            bool horizontal;
            bool vertical;
        } flip;
    } params;

    // Factory methods (Lua-friendly, avoid complex constructors)
    static PreprocessOp crop(uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
        PreprocessOp op;
        op.type = CROP;
        op.params.crop = {x, y, w, h};
        return op;
    }

    static PreprocessOp resize(uint32_t w, uint32_t h, bool keep_aspect = false) {
        PreprocessOp op;
        op.type = RESIZE;
        op.params.resize = {w, h, keep_aspect};
        return op;
    }

    static PreprocessOp pad(uint32_t top, uint32_t bottom, uint32_t left, uint32_t right, uint8_t value = 114) {
        PreprocessOp op;
        op.type = PAD;
        op.params.pad = {top, bottom, left, right, value};
        return op;
    }

    static PreprocessOp color(PIXEL_FORMAT_E format) {
        PreprocessOp op;
        op.type = COLOR;
        op.params.color = {format};
        return op;
    }

    static PreprocessOp rotate(int angle_degrees) {
        PreprocessOp op;
        op.type = ROTATE;
        op.params.rotate = {angle_degrees};
        return op;
    }

    static PreprocessOp flip(bool horizontal, bool vertical) {
        PreprocessOp op;
        op.type = FLIP;
        op.params.flip = {horizontal, vertical};
        return op;
    }
};

} // namespace lua_cv
