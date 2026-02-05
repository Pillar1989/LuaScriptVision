#pragma once

#include "node.h"
#include "message_box.h"
#include "pipeline_context.h"

#include <vector>
#include <mutex>

namespace node {

// Type alias for context-based MessageBox
using ContextBox = MessageBox<PipelineContext>;

// DataNode extends Node with frame input/output capabilities
class DataNode : public Node {
public:
    explicit DataNode(const std::string& id,
                      const std::string& type,
                      size_t inbox_capacity = 4)
        : Node(id, type), inbox_(inbox_capacity) {}

    // Get input queue (for upstream to attach)
    virtual ContextBox* inbox() { return &inbox_; }

    // Attach/detach downstream subscribers
    virtual void attach(ContextBox* subscriber);
    virtual void detach(ContextBox* subscriber);

protected:
    ContextBox inbox_;                            // Input queue
    std::vector<ContextBox*> downstream_;         // Downstream subscribers
    std::mutex subscribers_mutex_;
};

inline void DataNode::attach(ContextBox* subscriber) {
    if (!subscriber) return;
    std::lock_guard<std::mutex> lock(subscribers_mutex_);
    downstream_.push_back(subscriber);
}

inline void DataNode::detach(ContextBox* subscriber) {
    if (!subscriber) return;
    std::lock_guard<std::mutex> lock(subscribers_mutex_);
    downstream_.erase(
        std::remove(downstream_.begin(), downstream_.end(), subscriber),
        downstream_.end());
}

} // namespace node
