#pragma once

#include "cvi_frame.h"
#include "cv_types.h"

namespace lua_cv {

class CvProcessor {
public:
    virtual ~CvProcessor() = default;
    virtual void resize(Frame& frame, int width, int height) = 0;
    virtual void cvtColor(Frame& frame, ColorConversion code) = 0;
    virtual void crop(Frame& frame, int x, int y, int w, int h) = 0;
};

} // namespace lua_cv
