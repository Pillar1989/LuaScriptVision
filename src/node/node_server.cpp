#include "node_server.h"

#include <mosquitto.h>

namespace node {

NodeServer::NodeServer()
    : NodeServer(Config{}) {
}

NodeServer::NodeServer(const Config& config)
    : config_(config) {
    // Initialize topic prefixes
    topic_in_prefix_ = "sscma/v0/" + config_.client_id + "/node/in";
    topic_out_prefix_ = "sscma/v0/" + config_.client_id + "/node/out";

    // Initialize mosquitto library (safe to call multiple times)
    mosquitto_lib_init();
}

NodeServer::~NodeServer() {
    stop();
    mosquitto_lib_cleanup();
}

bool NodeServer::start() {
    if (running_.load(std::memory_order_acquire)) {
        return true;
    }

    // Create mosquitto client
    mosq_ = mosquitto_new(config_.client_id.c_str(), true, this);
    if (!mosq_) {
        return false;
    }

    // Set callbacks
    mosquitto_connect_callback_set(mosq_, on_connect_cb);
    mosquitto_disconnect_callback_set(mosq_, on_disconnect_cb);
    mosquitto_message_callback_set(mosq_, on_message_cb);

    // Set auto-reconnect parameters
    mosquitto_reconnect_delay_set(mosq_, 2, 30, true);

    // Set credentials if provided
    if (!config_.username.empty()) {
        mosquitto_username_pw_set(mosq_,
            config_.username.c_str(),
            config_.password.c_str());
    }

    // Connect to broker
    int rc = mosquitto_connect(mosq_,
        config_.host.c_str(),
        config_.port,
        config_.keep_alive);
    if (rc != MOSQ_ERR_SUCCESS) {
        mosquitto_destroy(mosq_);
        mosq_ = nullptr;
        return false;
    }

    // Create executor for async command processing
    executor_ = std::make_unique<Executor>("node_server");

    // Bind factory to server
    NodeFactory::instance().setServer(this);

    running_.store(true, std::memory_order_release);

    // Start mosquitto loop in separate thread
    loop_thread_ = std::thread([this]() {
        mosquitto_loop_forever(mosq_, -1, 1);
    });

    return true;
}

void NodeServer::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    running_.store(false, std::memory_order_release);

    // Stop mosquitto loop
    if (mosq_) {
        mosquitto_disconnect(mosq_);
    }

    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }

    // Cleanup
    if (executor_) {
        executor_->cancel();
        executor_.reset();
    }

    // Destroy all nodes
    NodeFactory::instance().destroyAll();
    NodeFactory::instance().setServer(nullptr);

    if (mosq_) {
        mosquitto_destroy(mosq_);
        mosq_ = nullptr;
    }

    connected_.store(false, std::memory_order_release);
}

void NodeServer::response(const std::string& node_id,
                          const std::string& name,
                          int code,
                          const nlohmann::json& data) {
    if (!connected_.load(std::memory_order_acquire)) {
        return;  // Silently drop if not connected
    }

    nlohmann::json msg;
    msg["type"] = static_cast<int>(MessageType::RESPONSE);
    msg["name"] = name;
    msg["code"] = code;
    msg["data"] = data;

    std::string topic = topic_out_prefix_ + "/" + node_id;
    std::string payload = msg.dump();

    mosquitto_publish(mosq_, nullptr, topic.c_str(),
                      static_cast<int>(payload.size()),
                      payload.data(), 0, false);
}

void NodeServer::event(const std::string& node_id,
                       const std::string& name,
                       int code,
                       const nlohmann::json& data) {
    if (!connected_.load(std::memory_order_acquire)) {
        return;  // Silently drop if not connected
    }

    nlohmann::json msg;
    msg["type"] = static_cast<int>(MessageType::EVENT);
    msg["name"] = name;
    msg["code"] = code;
    msg["data"] = data;

    std::string topic = topic_out_prefix_ + "/" + node_id;
    std::string payload = msg.dump();

    mosquitto_publish(mosq_, nullptr, topic.c_str(),
                      static_cast<int>(payload.size()),
                      payload.data(), 0, false);
}

void NodeServer::onConnect(int rc) {
    if (rc == MOSQ_ERR_SUCCESS) {
        connected_.store(true, std::memory_order_release);

        // Subscribe to input topics
        std::string topic = topic_in_prefix_ + "/+";
        mosquitto_subscribe(mosq_, nullptr, topic.c_str(), 0);
    }
}

void NodeServer::onDisconnect(int /*rc*/) {
    connected_.store(false, std::memory_order_release);
    // mosquitto_loop will auto-reconnect
}

void NodeServer::onMessage(const std::string& topic, const std::string& payload) {
    // Extract node_id from topic
    std::string node_id = extractNodeId(topic);
    if (node_id.empty()) {
        return;
    }

    // Parse JSON
    nlohmann::json msg;
    try {
        msg = nlohmann::json::parse(payload);
    } catch (...) {
        return;  // Invalid JSON
    }

    // Submit to executor for async processing
    executor_->submit([this, node_id, msg]() {
        handleRequest(node_id, msg);
        return false;  // Don't requeue
    });
}

void NodeServer::handleRequest(const std::string& node_id, const nlohmann::json& msg) {
    int type = msg.value("type", 0);
    if (type != static_cast<int>(MessageType::REQUEST)) {
        return;  // Ignore non-request messages
    }

    std::string action = msg.value("name", "");
    nlohmann::json data = msg.value("data", nlohmann::json::object());

    int code = MA_OK;
    nlohmann::json result;

    try {
        if (action == "create") {
            code = handleCreate(node_id, data);
            if (code != MA_OK) {
                result["error"] = NodeFactory::instance().lastErrorReason();
            }
        } else if (action == "destroy") {
            code = NodeFactory::instance().destroy(node_id);
        } else if (action == "start") {
            code = NodeFactory::instance().start(node_id);
        } else if (action == "stop") {
            code = NodeFactory::instance().stop(node_id);
        } else if (action == "list") {
            auto ids = NodeFactory::instance().list();
            result["nodes"] = ids;
        } else {
            // Control command
            code = NodeFactory::instance().control(node_id, action, data);
        }
    } catch (const std::exception& e) {
        code = MA_EINVAL;
        result["error"] = e.what();
    }

    response(node_id, action, code, result);
}

int NodeServer::handleCreate(const std::string& node_id, const nlohmann::json& data) {
    std::string type = data.value("type", "");
    if (type.empty()) {
        return MA_EINVAL;
    }

    nlohmann::json config = data.value("config", nlohmann::json::object());
    std::vector<std::string> deps;

    if (data.contains("dependencies")) {
        for (const auto& d : data["dependencies"]) {
            deps.push_back(d.get<std::string>());
        }
    }

    Node* node = NodeFactory::instance().create(node_id, type, config, deps);
    return node ? MA_OK : NodeFactory::instance().lastErrorCode();
}

std::string NodeServer::extractNodeId(const std::string& topic) {
    // Topic format: sscma/v0/<client>/node/in/<node_id>
    // Find last '/' and extract node_id
    size_t pos = topic.rfind('/');
    if (pos == std::string::npos || pos == topic.length() - 1) {
        return "";
    }
    return topic.substr(pos + 1);
}

// Static callbacks
void NodeServer::on_connect_cb(struct mosquitto* /*mosq*/, void* obj, int rc) {
    auto* server = static_cast<NodeServer*>(obj);
    server->onConnect(rc);
}

void NodeServer::on_disconnect_cb(struct mosquitto* /*mosq*/, void* obj, int rc) {
    auto* server = static_cast<NodeServer*>(obj);
    server->onDisconnect(rc);
}

void NodeServer::on_message_cb(struct mosquitto* /*mosq*/, void* obj,
                                const struct mosquitto_message* msg) {
    auto* server = static_cast<NodeServer*>(obj);
    std::string topic(msg->topic);
    std::string payload(static_cast<char*>(msg->payload), msg->payloadlen);
    server->onMessage(topic, payload);
}

} // namespace node
