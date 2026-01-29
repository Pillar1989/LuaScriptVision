#pragma once

#include <memory>

#include "input_source.h"

namespace lua_cv {

class CameraSource : public InputSource {
public:
    CameraSource();
    ~CameraSource() override;

    bool open(const std::string& source) override;
    bool read(Frame& frame) override;
    bool wait_for_ready(int timeout_ms);
    double fps() const;
    int last_error() const;
    void release(Frame& frame) override;
    void close() override;
    InputType type() const override { return InputType::CAMERA; }
    bool is_opened() const override { return opened_; }

private:
#ifdef USE_CVI_CAMERA
    std::unique_ptr<class CviCamera> camera_;
#endif
    bool opened_ = false;
};

} // namespace lua_cv
