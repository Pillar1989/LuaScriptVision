#pragma once

#include "shared_frame.h"

#include <nlohmann/json.hpp>

namespace node {

// PipelineContext carries frame data between nodes
// - Used by all DataNode types (CameraNode, ModelNode, etc.)
// - Move-only semantics to prevent double-release
// - Destructor automatically releases frame reference
struct PipelineContext {
    SharedFrame* frame;               // Frame data (reference counted, not owned)
    nlohmann::json upstream_result;   // Result from upstream node (empty = from CameraNode)
    uint64_t frame_id;                // Frame sequence number

    // Destructor releases frame reference
    ~PipelineContext() {
        if (frame) {
            frame->release();
            frame = nullptr;
        }
    }

    // Move-only: prevent copying to avoid double-release
    PipelineContext(const PipelineContext&) = delete;
    PipelineContext& operator=(const PipelineContext&) = delete;

    PipelineContext(PipelineContext&& other) noexcept
        : frame(other.frame),
          upstream_result(std::move(other.upstream_result)),
          frame_id(other.frame_id) {
        other.frame = nullptr;  // Transfer ownership
    }

    PipelineContext& operator=(PipelineContext&& other) noexcept {
        if (this != &other) {
            if (frame) {
                frame->release();  // Release current reference
            }
            frame = other.frame;
            upstream_result = std::move(other.upstream_result);
            frame_id = other.frame_id;
            other.frame = nullptr;  // Transfer ownership
        }
        return *this;
    }

    // Constructor
    // Note: Caller must call frame->ref() BEFORE creating PipelineContext
    // The constructor does NOT call ref() - it assumes caller has already done so
    PipelineContext(SharedFrame* f, nlohmann::json result, uint64_t id)
        : frame(f), upstream_result(std::move(result)), frame_id(id) {}

    // Check if this context came directly from CameraNode (no upstream result)
    bool is_from_camera() const {
        return upstream_result.empty() || upstream_result.is_null();
    }
};

} // namespace node
