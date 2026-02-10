#pragma once

#include <atomic>
#include <string>
#include <thread>

namespace lua_cv {

class PreviewHttpServer {
public:
    struct Config {
        int port = 8000;
        int ws_video_port = 8080;
        int ws_infer_port = 8090;
        int infer_width = 640;
        int infer_height = 640;
    };

    explicit PreviewHttpServer(const Config& config);
    ~PreviewHttpServer();

    PreviewHttpServer(const PreviewHttpServer&) = delete;
    PreviewHttpServer& operator=(const PreviewHttpServer&) = delete;
    PreviewHttpServer(PreviewHttpServer&&) = delete;
    PreviewHttpServer& operator=(PreviewHttpServer&&) = delete;

    bool start();
    void stop();
    bool is_running() const { return running_.load(std::memory_order_acquire); }

    std::string get_url(const std::string& host = "<device_ip>") const;

private:
    void run_loop();

#ifdef USE_MONGOOSE_WS
    struct Impl;
    Impl* impl_ = nullptr;
#endif

    Config config_;
    std::atomic<bool> running_{false};
    std::thread io_thread_;
};

}  // namespace lua_cv
