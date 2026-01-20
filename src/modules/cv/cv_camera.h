#pragma once

#include "cvi_frame.h"

namespace lua_cv {

class CvCamera {
public:
    virtual ~CvCamera() = default;

    virtual bool open() = 0;
    virtual bool read(Frame& frame) = 0;
    virtual void release() = 0;

    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual double fps() const = 0;
    virtual bool is_opened() const = 0;
};

} // namespace lua_cv
