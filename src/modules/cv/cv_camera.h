#pragma once

#include "cvi_frame.h"

namespace lua_cv {

/**
 * CvCamera - Abstract camera interface
 *
 * Provides a unified interface for camera capture that can be implemented by:
 * - CviCamera: Hardware camera using SG200X VI/VPSS pipeline (zero-copy)
 * - OpenCvCamera: Software camera using cv::VideoCapture (CPU)
 */
class CvCamera {
public:
    virtual ~CvCamera() = default;

    /**
     * Open camera and initialize capture pipeline
     * @return true on success, false on failure
     */
    virtual bool open() = 0;

    /**
     * Read one frame from camera
     * @param frame Output frame (modified in-place)
     * @return true if frame captured successfully, false otherwise
     */
    virtual bool read(Frame& frame) = 0;

    /**
     * Release camera resources and close capture
     */
    virtual void release() = 0;

    /**
     * Get camera frame width
     */
    virtual int width() const = 0;

    /**
     * Get camera frame height
     */
    virtual int height() const = 0;

    /**
     * Get camera frame rate (FPS)
     */
    virtual double fps() const = 0;

    /**
     * Check if camera is opened
     */
    virtual bool is_opened() const = 0;
};

} // namespace lua_cv
