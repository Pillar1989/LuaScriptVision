#include "node/node_server.h"

#include <atomic>
#include <csignal>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using node::NodeServer;

namespace {
std::atomic<bool> g_running{true};

void handle_signal(int) {
    g_running.store(false, std::memory_order_release);
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --host <host>         MQTT broker host (default: localhost)\n"
              << "  --port <port>         MQTT broker port (default: 1883)\n"
              << "  --client-id <id>      MQTT client id (default: recamera)\n"
              << "  --user <username>     MQTT username\n"
              << "  --password <password> MQTT password\n"
              << "  --keep-alive <sec>    MQTT keep-alive seconds (default: 60)\n"
              << "  -h, --help            Show this help\n";
}
}  // namespace

int main(int argc, char** argv) {
    NodeServer::Config config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
        } else if (arg == "--client-id" && i + 1 < argc) {
            config.client_id = argv[++i];
        } else if (arg == "--user" && i + 1 < argc) {
            config.username = argv[++i];
        } else if (arg == "--password" && i + 1 < argc) {
            config.password = argv[++i];
        } else if (arg == "--keep-alive" && i + 1 < argc) {
            config.keep_alive = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    NodeServer server(config);
    if (!server.start()) {
        std::cerr << "Failed to start node_server\n";
        return 1;
    }

    while (g_running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    server.stop();
    return 0;
}
