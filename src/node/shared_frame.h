#pragma once

#include "modules/cv/frame.h"

#include <atomic>
#include <memory>

namespace node {

class SharedFrame {
public:
    explicit SharedFrame(lua_cv::Frame&& frame)
        : frame_(std::make_unique<lua_cv::Frame>(std::move(frame))),
          ref_count_(1) {}

    // Reference counting
    void ref(int n = 1) {
        ref_count_.fetch_add(n, std::memory_order_relaxed);
    }

    void release() {
        if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Frame destructor handles VB release (VPSS/VB/VDEC owner)
            delete this;
        }
    }

    // Access underlying frame
    const lua_cv::Frame& frame() const { return *frame_; }
    lua_cv::Frame& frame() { return *frame_; }

    // Stats
    int ref_count() const { return ref_count_.load(std::memory_order_acquire); }
    uint64_t timestamp() const { return timestamp_; }
    void set_timestamp(uint64_t ts) { timestamp_ = ts; }
    uint64_t frame_id() const { return frame_id_; }
    void set_frame_id(uint64_t id) { frame_id_ = id; }

private:
    ~SharedFrame() = default;  // Only delete via release()

    std::unique_ptr<lua_cv::Frame> frame_;
    std::atomic<int> ref_count_;
    uint64_t timestamp_ = 0;
    uint64_t frame_id_ = 0;
};

} // namespace node
