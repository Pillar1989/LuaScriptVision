#pragma once

#include "cv_processor.h"

namespace lua_cv {

class OpenCvProcessor : public CvProcessor {
public:
    OpenCvProcessor() = default;
    ~OpenCvProcessor() override = default;

    void resize(Frame& frame, int width, int height) override;
    void cvtColor(Frame& frame, ColorConversion code) override;
    void crop(Frame& frame, int x, int y, int w, int h) override;
};

} // namespace lua_cv
