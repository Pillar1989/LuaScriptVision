#include "camera_source.h"

#include <stdexcept>

#ifdef USE_CVI_MPI
#include "mmf_context.h"
#endif

#ifdef USE_CVI_CAMERA
#include "cvi_camera.h"
#endif

namespace lua_cv {

CameraSource::CameraSource() = default;

CameraSource::~CameraSource() {
    close();
}

bool CameraSource::open(const std::string& source) {
    (void)source;
#ifndef USE_CVI_CAMERA
    throw std::runtime_error("CameraSource::open - USE_CVI_CAMERA not enabled");
#else
#ifdef USE_CVI_MPI
    if (!MmfContext::instance().is_initialized()) {
        throw std::runtime_error("CameraSource::open - MmfContext not initialized");
    }
#endif
    close();
    CviCamera::Config config;
    config.format = PixelFormat::BGR;
    config.enable_infer = true;
    camera_ = std::make_unique<CviCamera>(config);
    if (!camera_->open()) {
        camera_.reset();
        return false;
    }
    opened_ = true;
    return true;
#endif
}

bool CameraSource::read(Frame& frame) {
    if (!opened_) {
        throw std::runtime_error("CameraSource::read - source not opened");
    }
#ifndef USE_CVI_CAMERA
    (void)frame;
    throw std::runtime_error("CameraSource::read - USE_CVI_CAMERA not enabled");
#else
    return camera_->read(frame);
#endif
}

void CameraSource::release(Frame& frame) {
    frame.release();
}

void CameraSource::close() {
#ifdef USE_CVI_CAMERA
    if (camera_) {
        camera_->release();
        camera_.reset();
    }
#endif
    opened_ = false;
}

} // namespace lua_cv
