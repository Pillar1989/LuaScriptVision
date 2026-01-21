#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>

#include "inference_session.h"

#ifdef USE_ONNX_RUNTIME
#include "onnx_session.h"
#endif

#ifdef USE_CVI_TPU
#include "cvi_session.h"
#endif

namespace inference {

inline std::unique_ptr<InferenceSession> create_session(const std::string& model_path,
                                                        int num_threads = 4) {
    auto dot = model_path.find_last_of('.');
    std::string ext = (dot == std::string::npos) ? "" : model_path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (ext == "cvimodel") {
#ifdef USE_CVI_TPU
        return std::make_unique<CviSession>(model_path);
#else
        throw std::runtime_error("create_session: CVI Runtime not enabled");
#endif
    }

    if (ext == "onnx") {
#ifdef USE_ONNX_RUNTIME
        return std::make_unique<OnnxSession>(model_path, num_threads);
#else
        throw std::runtime_error("create_session: ONNX Runtime not enabled");
#endif
    }

    throw std::runtime_error("create_session: unsupported model format");
}

} // namespace inference
