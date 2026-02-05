/**
 * parallel_infer_stream - Node Framework based parallel inference + RTSP streaming
 *
 * Topology:
 *   CameraNode (camera) -> ModelNode (detector)
 *   CameraNode (camera) -> StreamNode (streamer)
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "node/node_factory.h"
#include "node/node_server.h"
#include "node/camera_node.h"
#include "node/model_node.h"
#include "node/stream_node.h"

namespace {
std::atomic<bool> g_stop{false};

void handle_signal(int) {
    g_stop.store(true, std::memory_order_release);
}

bool should_stop() {
    return g_stop.load(std::memory_order_acquire);
}

struct AppConfig {
    std::string script_path;
    std::string model_path;
    std::string input = "camera";
    std::string codec = "h264";
    std::string session = "live";
    std::string sensor = "ov5647";
    int rtsp_port = 554;
    int duration_sec = 0;
    int stream_fps = 30;
    int stream_bitrate_kbps = 4000;
    int stream_width = 1920;
    int stream_height = 1080;
    double infer_fps_limit = 0.0;
    float threshold = 0.25f;
    bool enable_mqtt = false;
    bool profile = false;

    // MQTT options
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_user;
    std::string mqtt_pass;
    std::string mqtt_client_id = "recamera";
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <script.lua> <model.cvimodel> [options]\n"
              << "\nOptions:\n"
              << "  --input SOURCE          Input source: camera (default: camera)\n"
              << "  --rtsp-port PORT        RTSP server port (default: 554)\n"
              << "  --codec h264|h265|mjpeg Video codec (default: h264)\n"
              << "  --session NAME          RTSP session name (default: live)\n"
              << "  --duration SECONDS      Run duration, 0=infinite (default: 0)\n"
              << "  --fps FPS               Stream FPS (default: 30)\n"
              << "  --bitrate KBPS          Stream bitrate in kbps (default: 4000)\n"
              << "  --width W               Stream width (default: 1920)\n"
              << "  --height H              Stream height (default: 1080)\n"
              << "  --infer-fps-limit FPS   Enable adaptive skip to limit inference FPS\n"
              << "  --sensor NAME           Sensor type (default: ov5647)\n"
              << "  --threshold VALUE       Confidence threshold (default: 0.25)\n"
              << "  --profile               Enable ModelNode profiling output\n"
              << "  --mqtt                  Enable MQTT events via NodeServer\n"
              << "  --mqtt-host HOST         MQTT broker host (default: localhost)\n"
              << "  --mqtt-port PORT         MQTT broker port (default: 1883)\n"
              << "  --mqtt-user USER         MQTT username\n"
              << "  --mqtt-pass PASS         MQTT password\n"
              << "  --mqtt-client-id ID      MQTT client id (default: recamera)\n"
              << "\nExamples:\n"
              << "  " << prog << " scripts/yolo11_tensor_detector.lua /userdata/Models/model.cvimodel --rtsp-port 8554\n"
              << "  " << prog << " scripts/yolo11_tensor_detector.lua /userdata/Models/model.cvimodel --mqtt\n";
}

bool parse_args(int argc, char* argv[], AppConfig* config) {
    if (argc < 3) {
        print_usage(argv[0]);
        return false;
    }

    config->script_path = argv[1];
    config->model_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            config->input = argv[++i];
        } else if (arg == "--rtsp-port" && i + 1 < argc) {
            config->rtsp_port = std::atoi(argv[++i]);
        } else if (arg == "--codec" && i + 1 < argc) {
            config->codec = argv[++i];
        } else if (arg == "--session" && i + 1 < argc) {
            config->session = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            config->duration_sec = std::atoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            config->stream_fps = std::atoi(argv[++i]);
        } else if (arg == "--bitrate" && i + 1 < argc) {
            config->stream_bitrate_kbps = std::atoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            config->stream_width = std::atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            config->stream_height = std::atoi(argv[++i]);
        } else if (arg == "--infer-fps-limit" && i + 1 < argc) {
            config->infer_fps_limit = std::atof(argv[++i]);
        } else if (arg == "--sensor" && i + 1 < argc) {
            config->sensor = argv[++i];
        } else if (arg == "--threshold" && i + 1 < argc) {
            config->threshold = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--profile") {
            config->profile = true;
        } else if (arg == "--mqtt") {
            config->enable_mqtt = true;
        } else if (arg == "--mqtt-host" && i + 1 < argc) {
            config->mqtt_host = argv[++i];
        } else if (arg == "--mqtt-port" && i + 1 < argc) {
            config->mqtt_port = std::atoi(argv[++i]);
        } else if (arg == "--mqtt-user" && i + 1 < argc) {
            config->mqtt_user = argv[++i];
        } else if (arg == "--mqtt-pass" && i + 1 < argc) {
            config->mqtt_pass = argv[++i];
        } else if (arg == "--mqtt-client-id" && i + 1 < argc) {
            config->mqtt_client_id = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

    if (config->input != "camera") {
        std::cerr << "Only camera input is supported in parallel_infer_stream.\n";
        return false;
    }

    return true;
}

bool start_node_or_fail(node::NodeFactory& factory,
                        const std::string& id,
                        const std::string& label) {
    int ret = factory.start(id);
    if (ret != node::MA_OK) {
        std::cerr << "[App] Failed to start " << label << " (" << id
                  << ") error=" << ret << "\n";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    AppConfig config;
    if (!parse_args(argc, argv, &config)) {
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::unique_ptr<node::NodeServer> server;
    if (config.enable_mqtt) {
        node::NodeServer::Config server_cfg;
        server_cfg.host = config.mqtt_host;
        server_cfg.port = config.mqtt_port;
        server_cfg.username = config.mqtt_user;
        server_cfg.password = config.mqtt_pass;
        server_cfg.client_id = config.mqtt_client_id;

        server = std::make_unique<node::NodeServer>(server_cfg);
        if (!server->start()) {
            std::cerr << "[App] MQTT connect failed, continuing without MQTT events.\n";
            server.reset();
        }
    }

    node::NodeFactory& factory = node::NodeFactory::instance();
    auto shutdown = [&](bool stop_server) {
        factory.stop("detector");
        factory.stop("streamer");
        factory.stop("camera");
        factory.destroyAll();
        if (stop_server && server) {
            server->stop();
        }
    };

    nlohmann::json camera_cfg = {
        {"width", config.stream_width},
        {"height", config.stream_height},
        {"fps", config.stream_fps},
        {"infer_fps_limit", config.infer_fps_limit},
        {"sensor", config.sensor},
        {"enable_stream", true},
        {"enable_inference", true}
    };

    if (!factory.create("camera", "camera", camera_cfg)) {
        std::cerr << "[App] Failed to create CameraNode: "
                  << factory.lastErrorReason() << "\n";
        shutdown(true);
        return 1;
    }

    nlohmann::json stream_cfg = {
        {"codec", config.codec},
        {"bitrate_kbps", config.stream_bitrate_kbps},
        {"fps", config.stream_fps},
        {"width", config.stream_width},
        {"height", config.stream_height},
        {"port", config.rtsp_port},
        {"session", config.session}
    };

    if (!factory.create("streamer", "stream", stream_cfg, {"camera"})) {
        std::cerr << "[App] Failed to create StreamNode: "
                  << factory.lastErrorReason() << "\n";
        shutdown(true);
        return 1;
    }

    nlohmann::json model_cfg = {
        {"model", config.model_path},
        {"script", config.script_path},
        {"threshold", config.threshold}
    };
    if (config.profile) {
        model_cfg["profile"] = true;
    }

    if (!factory.create("detector", "model", model_cfg, {"camera"})) {
        std::cerr << "[App] Failed to create ModelNode: "
                  << factory.lastErrorReason() << "\n";
        shutdown(true);
        return 1;
    }

    if (!start_node_or_fail(factory, "camera", "CameraNode") ||
        !start_node_or_fail(factory, "streamer", "StreamNode") ||
        !start_node_or_fail(factory, "detector", "ModelNode")) {
        shutdown(true);
        return 1;
    }

    std::cout << "[App] RTSP: rtsp://<device_ip>:" << config.rtsp_port
              << "/" << config.session << "\n";
    std::cout << "[App] Press Ctrl+C to stop...\n";

    auto start = std::chrono::steady_clock::now();
    while (!should_stop()) {
        if (config.duration_sec > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= config.duration_sec) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    shutdown(true);

    return 0;
}
