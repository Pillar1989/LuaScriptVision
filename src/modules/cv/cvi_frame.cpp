#include "cvi_frame.h"
#include <stdexcept>
#include <iostream>

#ifdef USE_CVI_MPI
#include <cvi_vpss.h>
#include <cvi_sys.h>
#endif

namespace lua_cv {

// ========== Constructors ==========

Frame::Frame()
    : data_()
    , type_(StorageType::EMPTY)
    , owns_memory_(false)
    , mat_cache_()
    , mat_cache_valid_(false)
#ifdef USE_CVI_MPI
    , vpss_grp_(-1)
    , vpss_chn_(-1)
#endif
    {
}

Frame::Frame(const cv::Mat& mat)
    : data_(mat.clone())  // Clone to own the data
    , type_(StorageType::OPENCV)
    , owns_memory_(true)
    , mat_cache_()
    , mat_cache_valid_(false)
#ifdef USE_CVI_MPI
    , vpss_grp_(-1)
    , vpss_chn_(-1)
#endif
    {
    if (mat.empty()) {
        type_ = StorageType::EMPTY;
        data_ = std::monostate{};
    }
}

#ifdef USE_CVI_MPI
Frame::Frame(const VIDEO_FRAME_INFO_S& frame, bool owns)
    : data_(frame)
    , type_(StorageType::VIDEO_FRAME)
    , owns_memory_(owns)
    , mat_cache_()
    , mat_cache_valid_(false)
    , vpss_grp_(-1)
    , vpss_chn_(-1) {
}

Frame::Frame(const VIDEO_FRAME_INFO_S& frame, VPSS_GRP grp, VPSS_CHN chn)
    : data_(frame)
    , type_(StorageType::VIDEO_FRAME)
    , owns_memory_(true)  // Always own when VPSS context is provided
    , mat_cache_()
    , mat_cache_valid_(false)
    , vpss_grp_(grp)
    , vpss_chn_(chn) {
}
#endif

Frame::~Frame() {
    release();
}

// ========== Move Semantics ==========

Frame::Frame(Frame&& other) noexcept
    : data_(std::move(other.data_))
    , type_(other.type_)
    , owns_memory_(other.owns_memory_)
    , mat_cache_(std::move(other.mat_cache_))
    , mat_cache_valid_(other.mat_cache_valid_)
#ifdef USE_CVI_MPI
    , vpss_grp_(other.vpss_grp_)
    , vpss_chn_(other.vpss_chn_)
#endif
    {
    // Reset other to empty state
    other.type_ = StorageType::EMPTY;
    other.owns_memory_ = false;
    other.mat_cache_valid_ = false;
#ifdef USE_CVI_MPI
    other.vpss_grp_ = -1;
    other.vpss_chn_ = -1;
#endif
}

Frame& Frame::operator=(Frame&& other) noexcept {
    if (this != &other) {
        // Release current resources
        release();

        // Move data
        data_ = std::move(other.data_);
        type_ = other.type_;
        owns_memory_ = other.owns_memory_;
        mat_cache_ = std::move(other.mat_cache_);
        mat_cache_valid_ = other.mat_cache_valid_;
#ifdef USE_CVI_MPI
        vpss_grp_ = other.vpss_grp_;
        vpss_chn_ = other.vpss_chn_;
#endif

        // Reset other
        other.type_ = StorageType::EMPTY;
        other.owns_memory_ = false;
        other.mat_cache_valid_ = false;
#ifdef USE_CVI_MPI
        other.vpss_grp_ = -1;
        other.vpss_chn_ = -1;
#endif
    }
    return *this;
}

// ========== Properties ==========

int Frame::width() const {
    switch (type_) {
        case StorageType::EMPTY:
            return 0;

        case StorageType::OPENCV: {
            const auto& mat = std::get<cv::Mat>(data_);
            return mat.cols;
        }

#ifdef USE_CVI_MPI
        case StorageType::VIDEO_FRAME: {
            const auto& frame = std::get<VIDEO_FRAME_INFO_S>(data_);
            return frame.stVFrame.u32Width;
        }
#endif

        default:
            return 0;
    }
}

int Frame::height() const {
    switch (type_) {
        case StorageType::EMPTY:
            return 0;

        case StorageType::OPENCV: {
            const auto& mat = std::get<cv::Mat>(data_);
            return mat.rows;
        }

#ifdef USE_CVI_MPI
        case StorageType::VIDEO_FRAME: {
            const auto& frame = std::get<VIDEO_FRAME_INFO_S>(data_);
            return frame.stVFrame.u32Height;
        }
#endif

        default:
            return 0;
    }
}

int Frame::channels() const {
    switch (type_) {
        case StorageType::EMPTY:
            return 0;

        case StorageType::OPENCV: {
            const auto& mat = std::get<cv::Mat>(data_);
            return mat.channels();
        }

#ifdef USE_CVI_MPI
        case StorageType::VIDEO_FRAME: {
            const auto& frame = std::get<VIDEO_FRAME_INFO_S>(data_);
            PIXEL_FORMAT_E fmt = frame.stVFrame.enPixelFormat;

            // Map CVI pixel format to channel count
            switch (fmt) {
                case PIXEL_FORMAT_RGB_888:
                case PIXEL_FORMAT_BGR_888:
                case PIXEL_FORMAT_RGB_888_PLANAR:
                case PIXEL_FORMAT_BGR_888_PLANAR:
                    return 3;

                case PIXEL_FORMAT_NV12:
                case PIXEL_FORMAT_NV21:
                    return 1;  // Luma plane (Y), chroma is separate

                case PIXEL_FORMAT_ARGB_8888:
                    return 4;

                default:
                    return 0;
            }
        }
#endif

        default:
            return 0;
    }
}

PixelFormat Frame::format() const {
    switch (type_) {
        case StorageType::EMPTY:
            return PixelFormat::UNKNOWN;

        case StorageType::OPENCV: {
            const auto& mat = std::get<cv::Mat>(data_);
            int cv_type = mat.type();

            // Map OpenCV type to PixelFormat
            if (cv_type == CV_8UC3) {
                return PixelFormat::BGR888;  // OpenCV default is BGR
            } else if (cv_type == CV_8UC1) {
                return PixelFormat::GRAY;
            } else {
                return PixelFormat::UNKNOWN;
            }
        }

#ifdef USE_CVI_MPI
        case StorageType::VIDEO_FRAME: {
            const auto& frame = std::get<VIDEO_FRAME_INFO_S>(data_);
            PIXEL_FORMAT_E fmt = frame.stVFrame.enPixelFormat;

            switch (fmt) {
                case PIXEL_FORMAT_RGB_888:
                case PIXEL_FORMAT_RGB_888_PLANAR:
                    return PixelFormat::RGB888;

                case PIXEL_FORMAT_BGR_888:
                case PIXEL_FORMAT_BGR_888_PLANAR:
                    return PixelFormat::BGR888;

                case PIXEL_FORMAT_NV12:
                    return PixelFormat::NV12;

                case PIXEL_FORMAT_NV21:
                    return PixelFormat::NV21;

                default:
                    return PixelFormat::UNKNOWN;
            }
        }
#endif

        default:
            return PixelFormat::UNKNOWN;
    }
}

// ========== Zero-Copy Support ==========

bool Frame::has_physical_addr() const {
#ifdef USE_CVI_MPI
    return type_ == StorageType::VIDEO_FRAME;
#else
    return false;
#endif
}

uint64_t Frame::physical_addr() const {
#ifdef USE_CVI_MPI
    if (type_ != StorageType::VIDEO_FRAME) {
        throw std::runtime_error("Frame::physical_addr() - frame is not VIDEO_FRAME type");
    }

    const auto& frame = std::get<VIDEO_FRAME_INFO_S>(data_);
    return frame.stVFrame.u64PhyAddr[0];
#else
    throw std::runtime_error("Frame::physical_addr() - CVI MPI not available");
#endif
}

size_t Frame::physical_size() const {
#ifdef USE_CVI_MPI
    if (type_ != StorageType::VIDEO_FRAME) {
        throw std::runtime_error("Frame::physical_size() - frame is not VIDEO_FRAME type");
    }

    const auto& frame = std::get<VIDEO_FRAME_INFO_S>(data_);
    return frame.stVFrame.u32Length[0];
#else
    throw std::runtime_error("Frame::physical_size() - CVI MPI not available");
#endif
}

// ========== Conversions ==========

const cv::Mat& Frame::to_mat() const {
    if (type_ == StorageType::EMPTY) {
        throw std::runtime_error("Frame::to_mat() - frame is empty");
    }

    // If already OpenCV backend, return directly
    if (type_ == StorageType::OPENCV) {
        return std::get<cv::Mat>(data_);
    }

#ifdef USE_CVI_MPI
    // VIDEO_FRAME → cv::Mat conversion (lazy, with caching)
    if (type_ == StorageType::VIDEO_FRAME) {
        if (!mat_cache_valid_) {
            const auto& frame = std::get<VIDEO_FRAME_INFO_S>(data_);
            mat_cache_ = video_frame_to_mat(frame);
            mat_cache_valid_ = true;
        }
        return mat_cache_;
    }
#endif

    throw std::runtime_error("Frame::to_mat() - unknown storage type");
}

#ifdef USE_CVI_MPI
const VIDEO_FRAME_INFO_S& Frame::to_video_frame() const {
    if (type_ != StorageType::VIDEO_FRAME) {
        throw std::runtime_error("Frame::to_video_frame() - frame is not VIDEO_FRAME type");
    }

    return std::get<VIDEO_FRAME_INFO_S>(data_);
}
#endif

// ========== Resource Management ==========

void Frame::release() {
    if (type_ == StorageType::EMPTY) {
        return;
    }

#ifdef USE_CVI_MPI
    if (type_ == StorageType::VIDEO_FRAME && owns_memory_) {
        release_video_frame();
    }
#endif

    // Clear data
    data_ = std::monostate{};
    type_ = StorageType::EMPTY;
    owns_memory_ = false;
    mat_cache_ = cv::Mat();
    mat_cache_valid_ = false;
}

Frame Frame::clone() const {
    switch (type_) {
        case StorageType::EMPTY:
            return Frame();

        case StorageType::OPENCV: {
            const auto& mat = std::get<cv::Mat>(data_);
            return Frame(mat);  // Constructor already clones
        }

#ifdef USE_CVI_MPI
        case StorageType::VIDEO_FRAME: {
            // For VIDEO_FRAME, convert to cv::Mat and create OpenCV-backed frame
            // (cloning VIDEO_FRAME_INFO_S requires complex VB pool operations)
            const cv::Mat& mat = to_mat();
            return Frame(mat);
        }
#endif

        default:
            return Frame();
    }
}

// ========== Internal Helpers ==========

#ifdef USE_CVI_MPI
cv::Mat Frame::video_frame_to_mat(const VIDEO_FRAME_INFO_S& frame) const {
    const VIDEO_FRAME_S& vf = frame.stVFrame;
    PIXEL_FORMAT_E fmt = vf.enPixelFormat;
    uint32_t width = vf.u32Width;
    uint32_t height = vf.u32Height;
    uint32_t stride = vf.u32Stride[0];

    // Get virtual address for CPU access
    void* virt_addr = CVI_SYS_Mmap(vf.u64PhyAddr[0], vf.u32Length[0]);
    if (virt_addr == nullptr) {
        throw std::runtime_error("Frame::video_frame_to_mat() - CVI_SYS_Mmap failed");
    }

    cv::Mat result;

    try {
        switch (fmt) {
            case PIXEL_FORMAT_RGB_888: {
                // Packed RGB888
                cv::Mat temp(height, width, CV_8UC3, virt_addr, stride);
                result = temp.clone();
                break;
            }

            case PIXEL_FORMAT_BGR_888: {
                // Packed BGR888
                cv::Mat temp(height, width, CV_8UC3, virt_addr, stride);
                result = temp.clone();
                break;
            }

            case PIXEL_FORMAT_NV12:
            case PIXEL_FORMAT_NV21: {
                // NV12/NV21: Y plane + UV plane
                // Y plane: height rows
                // UV plane: height/2 rows
                cv::Mat yuv(height * 3 / 2, width, CV_8UC1, virt_addr, stride);

                // Convert to BGR for easier processing
                int code = (fmt == PIXEL_FORMAT_NV12) ? cv::COLOR_YUV2BGR_NV12 : cv::COLOR_YUV2BGR_NV21;
                cv::cvtColor(yuv, result, code);
                break;
            }

            default: {
                CVI_SYS_Munmap(virt_addr, vf.u32Length[0]);
                throw std::runtime_error("Frame::video_frame_to_mat() - unsupported pixel format");
            }
        }
    } catch (...) {
        CVI_SYS_Munmap(virt_addr, vf.u32Length[0]);
        throw;
    }

    // Unmap memory
    CVI_SYS_Munmap(virt_addr, vf.u32Length[0]);

    return result;
}

void Frame::release_video_frame() {
    if (type_ != StorageType::VIDEO_FRAME) {
        return;
    }

    if (!owns_memory_) {
        return;
    }

    try {
        const auto& frame = std::get<VIDEO_FRAME_INFO_S>(data_);

        // If VPSS context is available, release via VPSS API
        if (vpss_grp_ >= 0 && vpss_chn_ >= 0) {
            CVI_S32 rc = CVI_VPSS_ReleaseChnFrame(vpss_grp_, vpss_chn_, &frame);
            if (rc != CVI_SUCCESS) {
                std::cerr << "[ERROR] Frame::release_video_frame() - "
                          << "CVI_VPSS_ReleaseChnFrame failed: " << rc
                          << " (GRP=" << vpss_grp_ << ", CHN=" << vpss_chn_ << ")" << std::endl;
            }
        }
        // No VPSS context - frame may be from VI or other source
        // In this case, we cannot safely release it via VPSS API
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Frame::release_video_frame() - " << e.what() << std::endl;
    }
}
#endif

} // namespace lua_cv
