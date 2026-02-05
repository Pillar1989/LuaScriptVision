#pragma once

#include "data_node.h"
#include "preprocess_config.h"
#include "luaref_json.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "LuaIntf.h"
#include <nlohmann/json.hpp>

// Forward declarations
namespace inference { class CviSession; }

namespace node {

class CameraNode;

// ModelNode: TPU inference with Lua postprocessing
class ModelNode : public DataNode {
public:
    enum InputMode {
        FULL_FRAME,      // Full frame + upstream metadata
        CROPPED_ROI      // VPSS crop from upstream boxes
    };

    ModelNode(const std::string& id, const std::string& type);
    ~ModelNode() override;

    // Node lifecycle
    int onCreate(const nlohmann::json& config) override;
    int onStart() override;
    int onStop() override;
    int onDestroy() override;
    int onControl(const std::string& action, const nlohmann::json& data) override;

    // Statistics
    uint64_t infer_count() const { return infer_count_.load(std::memory_order_relaxed); }
    uint64_t error_count() const { return error_count_.load(std::memory_order_relaxed); }
    double infer_ema_ms() const { return infer_ema_ms_; }

private:
    void inferLoop();
    nlohmann::json runInference(const lua_cv::Frame& frame, const nlohmann::json& upstream);
    nlohmann::json runFullFrameInference(const lua_cv::Frame& frame, const nlohmann::json& upstream);
    nlohmann::json runCroppedRoiInference(const lua_cv::Frame& frame, const nlohmann::json& upstream);
    nlohmann::json callPostprocess(
        std::vector<std::vector<float>> outputs,
        std::vector<std::vector<int64_t>> output_shapes,
        const nlohmann::json& meta);
    void forwardToDownstream(PipelineContext* ctx, const nlohmann::json& result);
    void updateEwma(double elapsed_ms);
    void cleanupLuaRef();
    void emitProfile(const nlohmann::json& profile);

    // Configuration
    std::string model_path_;
    std::string script_path_;
    InputMode input_mode_ = FULL_FRAME;
    int crop_width_ = 112;
    int crop_height_ = 112;
    bool crop_size_explicit_ = false;
    float conf_threshold_ = 0.25f;
    int infer_timeout_ms_ = 5000;

    // Lua integration (independent State for thread safety)
    lua_State* L_ = nullptr;
    LuaIntf::LuaRef postprocess_;
    LuaIntf::LuaRef select_rois_;
    LuaIntf::LuaRef preprocess_config_ref_;
    PreprocessConfig preprocess_config_;
    bool preprocess_config_explicit_ = false;
    bool profile_ = false;

    // TPU Session
    std::unique_ptr<inference::CviSession> session_;

    // Inference thread
    std::thread infer_thread_;
    std::atomic<bool> running_{false};

    // Upstream camera reference (for timing feedback)
    CameraNode* upstream_camera_ = nullptr;

    // Statistics
    std::atomic<uint64_t> infer_count_{0};
    std::atomic<uint64_t> error_count_{0};
    double infer_ema_ms_ = 0.0;
    static constexpr double kEmaAlpha = 0.2;
};

} // namespace node
