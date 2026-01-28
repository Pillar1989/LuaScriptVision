#pragma once

#include <string>

#include "frame.h"

namespace lua_cv {

enum class InputType {
    IMAGE = 0,
    VIDEO,
    CAMERA,
};

class InputSource {
public:
    virtual ~InputSource() = default;

    virtual bool open(const std::string& source) = 0;
    virtual bool read(Frame& frame) = 0;
    virtual void release(Frame& frame) = 0;
    virtual void close() = 0;
    virtual InputType type() const = 0;
    virtual bool is_opened() const = 0;
};

} // namespace lua_cv
