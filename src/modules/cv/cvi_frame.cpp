#include "cvi_frame.h"

#include <cstring>
#include <stdexcept>

#ifdef USE_CVI_MPI
#include <cvi_sys.h>
#include <cvi_vdec.h>
#endif

namespace lua_cv {

Frame::Frame() = default;

Frame::Frame(const cv::Mat& mat) : mat_(mat) {
    if (!mat_.empty()) {
        storage_type_ = StorageType::OPENCV;
        format_ = PixelFormat::BGR;
    }
}

Frame::Frame(const cv::Mat& mat, PixelFormat format) : mat_(mat) {
    if (!mat_.empty()) {
        storage_type_ = StorageType::OPENCV;
        format_ = format;
    }
}

Frame::Frame(cv::Mat&& mat) : mat_(std::move(mat)) {
    if (!mat_.empty()) {
        storage_type_ = StorageType::OPENCV;
        format_ = PixelFormat::BGR;
    }
}

Frame::Frame(cv::Mat&& mat, PixelFormat format) : mat_(std::move(mat)) {
    if (!mat_.empty()) {
        storage_type_ = StorageType::OPENCV;
        format_ = format;
    }
}

#ifdef USE_CVI_MPI
Frame::Frame(const VIDEO_FRAME_INFO_S& frame, bool owns_memory)
    : cvi_frame_(frame), owns_vb_block_(owns_memory) {
    storage_type_ = StorageType::CVI;
    format_ = from_cvi_pixel_format(frame.stVFrame.enPixelFormat);
    if (owns_vb_block_) {
        vb_block_ = CVI_VB_PhysAddr2Handle(frame.stVFrame.u64PhyAddr[0]);
    }
}

Frame::Frame(const VIDEO_FRAME_INFO_S& frame, int vpss_grp, int vpss_chn)
    : cvi_frame_(frame) {
    storage_type_ = StorageType::CVI;
    format_ = from_cvi_pixel_format(frame.stVFrame.enPixelFormat);
    vpss_grp_ = vpss_grp;
    vpss_chn_ = vpss_chn;
    owns_vpss_frame_ = false;
}
#endif

Frame::Frame(const Frame& other) {
    *this = other.clone();
}

Frame::Frame(Frame&& other) noexcept {
    *this = std::move(other);
}

Frame& Frame::operator=(const Frame& other) {
    if (this != &other) {
        *this = other.clone();
    }
    return *this;
}

Frame& Frame::operator=(Frame&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    release();

    storage_type_ = other.storage_type_;
    format_ = other.format_;
    mat_ = std::move(other.mat_);
    mat_cache_ = std::move(other.mat_cache_);
    mat_cache_valid_ = other.mat_cache_valid_;
    image_ = other.image_;

#ifdef USE_CVI_MPI
    cvi_frame_ = other.cvi_frame_;
    owns_vpss_frame_ = other.owns_vpss_frame_;
    owns_vb_block_ = other.owns_vb_block_;
    owns_vdec_frame_ = other.owns_vdec_frame_;
    vpss_grp_ = other.vpss_grp_;
    vpss_chn_ = other.vpss_chn_;
    vdec_chn_ = other.vdec_chn_;
    vb_block_ = other.vb_block_;

    other.owns_vpss_frame_ = false;
    other.owns_vb_block_ = false;
    other.owns_vdec_frame_ = false;
    other.vpss_grp_ = -1;
    other.vpss_chn_ = -1;
    other.vdec_chn_ = -1;
    other.vb_block_ = VB_INVALID_HANDLE;
#endif

    other.storage_type_ = StorageType::NONE;
    other.format_ = PixelFormat::UNKNOWN;
    other.mat_cache_valid_ = false;

    return *this;
}

Frame::~Frame() {
    release();
}

bool Frame::empty() const {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_.empty();
    }
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        return cvi_frame_.stVFrame.u32Width == 0 || cvi_frame_.stVFrame.u32Height == 0;
    }
#endif
    return true;
}

int Frame::width() const {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_.cols;
    }
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        return static_cast<int>(cvi_frame_.stVFrame.u32Width);
    }
#endif
    return 0;
}

int Frame::height() const {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_.rows;
    }
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        return static_cast<int>(cvi_frame_.stVFrame.u32Height);
    }
#endif
    return 0;
}

int Frame::channels() const {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_.channels();
    }
    return channels_for_format(format_);
}

bool Frame::has_physical_addr() const {
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        return cvi_frame_.stVFrame.u64PhyAddr[0] != 0;
    }
#endif
    return false;
}

uint64_t Frame::physical_addr() const {
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        return static_cast<uint64_t>(cvi_frame_.stVFrame.u64PhyAddr[0]);
    }
#endif
    return 0;
}

const cv::Mat& Frame::to_mat() const {
    return const_cast<Frame*>(this)->to_mat();
}

cv::Mat& Frame::to_mat() {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_;
    }
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        if (mat_cache_valid_) {
            return mat_cache_;
        }
        if (cvi_frame_.stVFrame.u64PhyAddr[0] == 0) {
            throw std::runtime_error("Frame::to_mat - CVI frame missing physical address");
        }

        size_t total_size = static_cast<size_t>(cvi_frame_.stVFrame.u32Length[0]) +
                            static_cast<size_t>(cvi_frame_.stVFrame.u32Length[1]) +
                            static_cast<size_t>(cvi_frame_.stVFrame.u32Length[2]);
        if (total_size == 0) {
            total_size = static_cast<size_t>(cvi_frame_.stVFrame.u32Stride[0]) *
                         static_cast<size_t>(cvi_frame_.stVFrame.u32Height);
        }

        void* mapped = CVI_SYS_MmapCache(cvi_frame_.stVFrame.u64PhyAddr[0],
                                         static_cast<CVI_U32>(total_size));
        if (!mapped) {
            throw std::runtime_error("Frame::to_mat - CVI_SYS_MmapCache failed");
        }
        CVI_SYS_IonInvalidateCache(cvi_frame_.stVFrame.u64PhyAddr[0], mapped,
                                   static_cast<CVI_U32>(total_size));

        uint8_t* base = static_cast<uint8_t*>(mapped);
        int w = static_cast<int>(cvi_frame_.stVFrame.u32Width);
        int h = static_cast<int>(cvi_frame_.stVFrame.u32Height);
        int stride = static_cast<int>(cvi_frame_.stVFrame.u32Stride[0]);

        if (format_ == PixelFormat::BGR || format_ == PixelFormat::RGB) {
            cv::Mat view(h, w, CV_8UC3, base, stride);
            mat_cache_ = view.clone();
        } else if (format_ == PixelFormat::GRAY) {
            cv::Mat view(h, w, CV_8UC1, base, stride);
            mat_cache_ = view.clone();
        } else if (format_ == PixelFormat::NV12 || format_ == PixelFormat::NV21) {
            cv::Mat yuv(h + h / 2, w, CV_8UC1, base, stride);
            int code = (format_ == PixelFormat::NV21) ? cv::COLOR_YUV2BGR_NV21
                                                      : cv::COLOR_YUV2BGR_NV12;
            cv::cvtColor(yuv, mat_cache_, code);
        } else {
            CVI_SYS_Munmap(mapped, static_cast<CVI_U32>(total_size));
            throw std::runtime_error("Frame::to_mat - unsupported CVI pixel format");
        }

        CVI_SYS_Munmap(mapped, static_cast<CVI_U32>(total_size));
        mat_cache_valid_ = true;
        return mat_cache_;
    }
#endif
    return mat_;
}

Frame Frame::clone() const {
    if (storage_type_ == StorageType::OPENCV) {
        return Frame(mat_.clone());
    }
    if (storage_type_ == StorageType::NONE) {
        return Frame();
    }
    cv::Mat cloned = to_mat().clone();
    return Frame(std::move(cloned));
}

void Frame::release() {
#ifdef USE_CVI_MPI
    if (owns_vdec_frame_) {
        if (vdec_chn_ >= 0) {
            CVI_VDEC_ReleaseFrame(static_cast<VDEC_CHN>(vdec_chn_), &cvi_frame_);
        }
        owns_vdec_frame_ = false;
    }

    if (owns_vpss_frame_) {
        if (vpss_grp_ >= 0 && vpss_chn_ >= 0) {
            CVI_VPSS_ReleaseChnFrame(static_cast<VPSS_GRP>(vpss_grp_),
                                     static_cast<VPSS_CHN>(vpss_chn_),
                                     &cvi_frame_);
        }
        owns_vpss_frame_ = false;
    }

    if (owns_vb_block_) {
        if (vb_block_ != VB_INVALID_HANDLE) {
            CVI_VB_ReleaseBlock(vb_block_);
        }
        owns_vb_block_ = false;
    }
#endif

    reset();
}

#ifdef USE_CVI_MPI
const VIDEO_FRAME_INFO_S* Frame::video_frame() const {
    if (storage_type_ != StorageType::CVI) {
        return nullptr;
    }
    return &cvi_frame_;
}

VIDEO_FRAME_INFO_S* Frame::video_frame() {
    if (storage_type_ != StorageType::CVI) {
        return nullptr;
    }
    return &cvi_frame_;
}

void Frame::set_vpss_owner(int vpss_grp, int vpss_chn) {
    vpss_grp_ = vpss_grp;
    vpss_chn_ = vpss_chn;
    owns_vpss_frame_ = true;
    owns_vb_block_ = false;
    vb_block_ = VB_INVALID_HANDLE;
    owns_vdec_frame_ = false;
    vdec_chn_ = -1;
}

void Frame::set_vb_owner(VB_BLK vb_block) {
    vb_block_ = vb_block;
    owns_vb_block_ = true;
    owns_vpss_frame_ = false;
    vpss_grp_ = -1;
    vpss_chn_ = -1;
    owns_vdec_frame_ = false;
    vdec_chn_ = -1;
}

void Frame::set_vdec_owner(int vdec_chn) {
    vdec_chn_ = vdec_chn;
    owns_vdec_frame_ = true;
    owns_vpss_frame_ = false;
    vpss_grp_ = -1;
    vpss_chn_ = -1;
    owns_vb_block_ = false;
    vb_block_ = VB_INVALID_HANDLE;
}
#endif

void Frame::reset() {
    mat_.release();
    mat_cache_.release();
    mat_cache_valid_ = false;
    storage_type_ = StorageType::NONE;
    format_ = PixelFormat::UNKNOWN;
#ifdef USE_CVI_MPI
    std::memset(&cvi_frame_, 0, sizeof(cvi_frame_));
    vpss_grp_ = -1;
    vpss_chn_ = -1;
    vdec_chn_ = -1;
    vb_block_ = VB_INVALID_HANDLE;
    owns_vpss_frame_ = false;
    owns_vb_block_ = false;
    owns_vdec_frame_ = false;
#endif
}

} // namespace lua_cv
