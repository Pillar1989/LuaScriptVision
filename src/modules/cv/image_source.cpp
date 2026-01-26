#include "image_source.h"

#include <fstream>
#include <stdexcept>

#ifdef USE_CVI_MPI
#include "hw_jpeg_decoder.h"
#include "mmf_context.h"
#endif

namespace lua_cv {

namespace {
#ifdef USE_CVI_MPI
HwJpegDecoder& decoder_instance() {
    static HwJpegDecoder decoder;
    return decoder;
}
#endif
}  // namespace

ImageSource::ImageSource() = default;

ImageSource::~ImageSource() {
    close();
}

bool ImageSource::open(const std::string& source) {
    if (source.empty()) {
        throw std::invalid_argument("ImageSource::open - source path is empty");
    }
#ifdef USE_CVI_MPI
    if (!MmfContext::instance().is_initialized()) {
        throw std::runtime_error("ImageSource::open - MmfContext not initialized");
    }
#endif

    close();
    if (!load_file(source)) {
        return false;
    }
    path_ = source;
    opened_ = true;
    return true;
}

bool ImageSource::read(Frame& frame) {
    if (!opened_) {
        throw std::runtime_error("ImageSource::read - source not opened");
    }
#ifndef USE_CVI_MPI
    (void)frame;
    throw std::runtime_error("ImageSource::read - USE_CVI_MPI not enabled");
#else
    if (data_.empty()) {
        throw std::runtime_error("ImageSource::read - no image data loaded");
    }
    if (!is_jpeg_format(data_.data(), data_.size())) {
        throw std::runtime_error("ImageSource::read - only JPEG supported");
    }

    HwJpegDecoder& decoder = decoder_instance();
    if (!decoder.ensure_initialized(width_, height_)) {
        throw std::runtime_error("ImageSource::read - VDEC init failed");
    }

    VIDEO_FRAME_INFO_S decoded = decoder.decode_sync(data_.data(), data_.size());
    Frame out(decoded, false);
    out.set_vdec_owner(decoder.channel());
    frame = std::move(out);
    return true;
#endif
}

void ImageSource::release(Frame& frame) {
    frame.release();
}

void ImageSource::close() {
#ifdef USE_CVI_MPI
    decoder_instance().cleanup();
#endif
    data_.clear();
    path_.clear();
    width_ = 0;
    height_ = 0;
    opened_ = false;
}

bool ImageSource::load_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("ImageSource::load_file - cannot open file: " + path);
    }

    std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("ImageSource::load_file - empty file: " + path);
    }
    file.seekg(0, std::ios::beg);

    data_.assign(static_cast<size_t>(size), 0);
    if (!file.read(reinterpret_cast<char*>(data_.data()), size)) {
        throw std::runtime_error("ImageSource::load_file - failed to read file: " + path);
    }

    if (!is_jpeg_format(data_.data(), data_.size())) {
        throw std::runtime_error("ImageSource::load_file - only JPEG supported");
    }

    uint32_t jpeg_width = 0;
    uint32_t jpeg_height = 0;
    if (!get_jpeg_dimensions(data_.data(), data_.size(), jpeg_width, jpeg_height)) {
        throw std::runtime_error("ImageSource::load_file - failed to parse JPEG header");
    }
    width_ = jpeg_width;
    height_ = jpeg_height;
#ifdef USE_CVI_MPI
    if (!MmfContext::is_supported_image_size(width_, height_)) {
        throw std::runtime_error("ImageSource::load_file - unsupported image resolution " +
                                 std::to_string(width_) + "x" + std::to_string(height_));
    }
#endif
    return true;
}

bool ImageSource::is_jpeg_format(const uint8_t* data, size_t size) const {
    if (!data || size < 4) {
        return false;
    }
    return data[0] == 0xFF && data[1] == 0xD8 &&
           data[size - 2] == 0xFF && data[size - 1] == 0xD9;
}

bool ImageSource::get_jpeg_dimensions(const uint8_t* data, size_t size,
                                      uint32_t& width, uint32_t& height) const {
    width = 0;
    height = 0;
    if (!data || size < 4 || data[0] != 0xFF || data[1] != 0xD8) {
        return false;
    }

    size_t i = 2;
    while (i + 1 < size) {
        if (data[i] != 0xFF) {
            ++i;
            continue;
        }

        while (i < size && data[i] == 0xFF) {
            ++i;
        }
        if (i >= size) {
            break;
        }

        uint8_t marker = data[i++];
        if (marker == 0xD9 || marker == 0xDA) {
            break;
        }
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            continue;
        }
        if (i + 1 >= size) {
            break;
        }

        uint16_t segment_len = static_cast<uint16_t>(data[i] << 8 | data[i + 1]);
        if (segment_len < 2 || i + segment_len > size) {
            break;
        }

        bool is_sof = (marker == 0xC0 || marker == 0xC1 || marker == 0xC2 ||
                       marker == 0xC3 || marker == 0xC5 || marker == 0xC6 ||
                       marker == 0xC7 || marker == 0xC9 || marker == 0xCA ||
                       marker == 0xCB || marker == 0xCD || marker == 0xCE ||
                       marker == 0xCF);
        if (is_sof) {
            if (segment_len < 7) {
                return false;
            }
            height = static_cast<uint32_t>(data[i + 3] << 8 | data[i + 4]);
            width = static_cast<uint32_t>(data[i + 5] << 8 | data[i + 6]);
            return width > 0 && height > 0;
        }

        i += segment_len;
    }

    return false;
}

} // namespace lua_cv
