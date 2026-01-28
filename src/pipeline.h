#pragma once

#include <memory>
#include <string>
#include <vector>

#include "LuaIntf.h"
#include "modules/lua_nn.h"

#ifdef USE_CVI_TPU
#include "inference/cvi_session.h"
#endif

#ifdef USE_CVI_MPI
#include "modules/cv/frame.h"
#include "modules/cv/input_source.h"
#include "modules/cv/cvi_vpss_processor.h"
#include "modules/cv/preprocess_config.h"
#endif

namespace pipeline {

struct StageTimings {
    double input_ms = 0.0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
};

struct InferenceResult {
    LuaIntf::LuaRef detections;
    StageTimings timings;
};

#if defined(USE_CVI_TPU) && defined(USE_CVI_MPI)

// Pipeline for .cvimodel inference on SG200X hardware.
// Manages input source → VPSS preprocessing → TPU inference → Lua postprocess.
class CviPipeline {
public:
    CviPipeline();
    ~CviPipeline();

    // Initialize the pipeline with a model and Lua script.
    // model_path: path to .cvimodel file
    // L: Lua state with postprocess function loaded
    // postprocess: Lua postprocess function reference
    // preprocess_config: optional Lua table with preprocess config
    void init(const std::string& model_path,
              lua_State* L,
              const LuaIntf::LuaRef& postprocess,
              const LuaIntf::LuaRef& preprocess_config);

    // Run inference on an image file.
    InferenceResult run_image(const std::string& image_path);

    // Run inference on a camera frame.
    InferenceResult run_camera();

    // Access the underlying session for metadata queries.
    inference::CviSession* session() const { return session_.get(); }

    const StageTimings& last_timings() const { return last_timings_; }

private:
    // Build VPSS preprocess plan from model input spec and config.
    void build_preprocess_plan();

    // Preprocess a frame via VPSS for TPU inference.
    // Returns the preprocessed frame and letterbox meta.
    lua_cv::Frame preprocess_frame(lua_cv::Frame& input_frame,
                                   LuaIntf::LuaRef* meta_out);

    // Run TPU inference on a preprocessed frame and return output tensors as Lua table.
    LuaIntf::LuaRef run_tpu(const lua_cv::Frame& preprocessed_frame);

    std::unique_ptr<inference::CviSession> session_;
    lua_State* L_ = nullptr;
    LuaIntf::LuaRef postprocess_;
    LuaIntf::LuaRef preprocess_config_lua_;

    // Preprocess plan derived from model input spec
    uint32_t model_input_w_ = 0;
    uint32_t model_input_h_ = 0;
    PIXEL_FORMAT_E model_input_format_ = PIXEL_FORMAT_MAX;
    bool letterbox_enabled_ = true;
    uint8_t pad_value_ = 114;

    // VPSS processor for preprocessing
    lua_cv::CviVpssProcessor vpss_processor_;

    StageTimings last_timings_{};
};

#endif  // USE_CVI_TPU && USE_CVI_MPI

}  // namespace pipeline
