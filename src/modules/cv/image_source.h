#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "input_source.h"

namespace lua_cv {

class ImageSource : public InputSource {
public:
    ImageSource();
    ~ImageSource() override;

    bool open(const std::string& source) override;
    bool read(Frame& frame) override;
    void release(Frame& frame) override;
    void close() override;
    InputType type() const override { return InputType::IMAGE; }
    bool is_opened() const override { return opened_; }

private:
    bool load_file(const std::string& path);
    bool is_jpeg_format(const uint8_t* data, size_t size) const;
    bool get_jpeg_dimensions(const uint8_t* data, size_t size,
                             uint32_t& width, uint32_t& height) const;

    std::string path_;
    std::vector<uint8_t> data_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    bool opened_ = false;
};

} // namespace lua_cv
