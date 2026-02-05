#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>

#include "node/node_server.h"
#include "node/node_factory.h"
#include "node/error_codes.h"

#include <mosquitto.h>

using namespace node;

// Test Node implementation for MQTT tests
class MqttTestNode : public Node {
public:
    MqttTestNode(const std::string& id, const std::string& type)
        : Node(id, type), param_(0) {}

    int onCreate(const nlohmann::json& config) override {
        if (config.contains("fail")) {
            return MA_EINVAL;
        }
        return MA_OK;
    }

    int onStart() override { return MA_OK; }
    int onStop() override { return MA_OK; }
    int onDestroy() override { return MA_OK; }

    int onControl(const std::string& action, const nlohmann::json& data) override {
        if (action == "set_param") {
            param_ = data.value("value", 0);
            return MA_OK;
        }
        if (action == "get_status") {
            response("get_status", MA_OK, {{"state", isStarted() ? "started" : "created"}});
            return MA_OK;
        }
        return MA_EINVAL;
    }

    int getParam() const { return param_; }

private:
    int param_;
};

// ============================================================================
// NodeServer Unit Tests (no broker required)
// ============================================================================

TEST(NodeServer, ConfigDefaults) {
    NodeServer::Config config;
    EXPECT_EQ(config.host, "localhost");
    EXPECT_EQ(config.port, 1883);
    EXPECT_EQ(config.client_id, "recamera");
    EXPECT_EQ(config.keep_alive, 60);
}

TEST(NodeServer, CustomConfig) {
    NodeServer::Config config;
    config.host = "192.168.42.1";
    config.port = 8883;
    config.client_id = "test_client";
    config.username = "user";
    config.password = "pass";

    NodeServer server(config);
    EXPECT_EQ(server.config().host, "192.168.42.1");
    EXPECT_EQ(server.config().port, 8883);
    EXPECT_EQ(server.config().client_id, "test_client");
}

TEST(NodeServer, InitialState) {
    NodeServer::Config config;
    config.client_id = "test";

    NodeServer server(config);
    EXPECT_FALSE(server.isRunning());
    EXPECT_FALSE(server.isConnected());
}

// ============================================================================
// MQTT Integration Tests (require broker)
// These tests are enabled by setting ENABLE_MQTT_TESTS=1 environment variable
// ============================================================================

class MqttIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if MQTT tests are enabled
        const char* env = std::getenv("ENABLE_MQTT_TESTS");
        if (!env || std::string(env) != "1") {
            GTEST_SKIP() << "MQTT integration tests disabled. Set ENABLE_MQTT_TESTS=1 to enable.";
        }

        // Register test node type
        NodeFactory::instance().registerNode("mqtt_test",
            [](const std::string& id, const std::string& type) {
                return std::make_unique<MqttTestNode>(id, type);
            });

        // Configure server
        config_.host = "localhost";
        config_.port = 1883;
        config_.client_id = "test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        if (const char* host = std::getenv("MQTT_TEST_HOST")) {
            if (*host) config_.host = host;
        }
        if (const char* port = std::getenv("MQTT_TEST_PORT")) {
            int port_val = std::atoi(port);
            if (port_val > 0) config_.port = port_val;
        }
        if (const char* user = std::getenv("MQTT_TEST_USER")) {
            config_.username = user;
        }
        if (const char* pass = std::getenv("MQTT_TEST_PASS")) {
            config_.password = pass;
        }
    }

    void TearDown() override {
        NodeFactory::instance().destroyAll();
    }

    NodeServer::Config config_;
};

// Helper class for MQTT client operations
class MqttTestClient {
public:
    MqttTestClient(const std::string& host, int port, const std::string& client_id)
        : connected_(false), message_received_(false) {
        mosquitto_lib_init();
        mosq_ = mosquitto_new(client_id.c_str(), true, this);

        mosquitto_connect_callback_set(mosq_, [](struct mosquitto*, void* obj, int rc) {
            auto* self = static_cast<MqttTestClient*>(obj);
            self->connected_ = (rc == MOSQ_ERR_SUCCESS);
        });

        mosquitto_message_callback_set(mosq_, [](struct mosquitto*, void* obj, const struct mosquitto_message* msg) {
            auto* self = static_cast<MqttTestClient*>(obj);
            std::lock_guard<std::mutex> lock(self->mutex_);
            self->last_topic_ = msg->topic;
            self->last_payload_ = std::string(static_cast<char*>(msg->payload), msg->payloadlen);
            self->message_received_ = true;
            self->cv_.notify_all();
        });

        host_ = host;
        port_ = port;
    }

    ~MqttTestClient() {
        if (mosq_) {
            mosquitto_disconnect(mosq_);
            mosquitto_destroy(mosq_);
        }
        mosquitto_lib_cleanup();
    }

    bool connect() {
        int rc = mosquitto_connect(mosq_, host_.c_str(), port_, 60);
        if (rc != MOSQ_ERR_SUCCESS) return false;

        // Start loop in background
        mosquitto_loop_start(mosq_);

        // Wait for connection
        for (int i = 0; i < 50 && !connected_; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return connected_;
    }

    void disconnect() {
        mosquitto_loop_stop(mosq_, true);
        mosquitto_disconnect(mosq_);
        connected_ = false;
    }

    void subscribe(const std::string& topic) {
        mosquitto_subscribe(mosq_, nullptr, topic.c_str(), 0);
    }

    void publish(const std::string& topic, const std::string& payload) {
        mosquitto_publish(mosq_, nullptr, topic.c_str(),
                         static_cast<int>(payload.size()), payload.data(), 0, false);
    }

    std::optional<std::string> waitMessage(const std::string& /*topic*/, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        message_received_ = false;

        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [this]() { return message_received_.load(); })) {
            return last_payload_;
        }
        return std::nullopt;
    }

private:
    struct mosquitto* mosq_;
    std::string host_;
    int port_;
    std::atomic<bool> connected_;
    std::atomic<bool> message_received_;
    std::string last_topic_;
    std::string last_payload_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

TEST_F(MqttIntegrationTest, StartStop) {
    NodeServer server(config_);

    EXPECT_TRUE(server.start());
    EXPECT_TRUE(server.isRunning());

    // Wait for connection
    for (int i = 0; i < 50 && !server.isConnected(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    EXPECT_TRUE(server.isConnected());

    server.stop();
    EXPECT_FALSE(server.isRunning());
}

TEST_F(MqttIntegrationTest, CreateRequest) {
    NodeServer server(config_);
    ASSERT_TRUE(server.start());

    // Wait for connection
    for (int i = 0; i < 50 && !server.isConnected(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::string client_id = "client_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    MqttTestClient client(config_.host, config_.port, client_id);
    ASSERT_TRUE(client.connect());

    // Subscribe to output topic
    std::string out_topic = "sscma/v0/" + config_.client_id + "/node/out/node1";
    client.subscribe(out_topic);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send create request
    nlohmann::json request = {
        {"type", 3},
        {"name", "create"},
        {"data", {
            {"type", "mqtt_test"},
            {"config", {{"key", "value"}}}
        }}
    };

    std::string in_topic = "sscma/v0/" + config_.client_id + "/node/in/node1";
    client.publish(in_topic, request.dump());

    // Wait for response
    auto response = client.waitMessage(out_topic, 2000);
    EXPECT_TRUE(response.has_value());

    if (response.has_value()) {
        auto msg = nlohmann::json::parse(response.value());
        EXPECT_EQ(msg["type"], 1);
        EXPECT_EQ(msg["name"], "create");
        EXPECT_EQ(msg["code"], 0);
    }

    server.stop();
}

TEST_F(MqttIntegrationTest, DestroyRequest) {
    NodeServer server(config_);
    ASSERT_TRUE(server.start());

    // Wait for connection
    for (int i = 0; i < 50 && !server.isConnected(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Create node first via factory
    NodeFactory::instance().create("node1", "mqtt_test", {});

    std::string client_id = "client_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    MqttTestClient client(config_.host, config_.port, client_id);
    ASSERT_TRUE(client.connect());

    std::string out_topic = "sscma/v0/" + config_.client_id + "/node/out/node1";
    client.subscribe(out_topic);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send destroy request
    nlohmann::json request = {
        {"type", 3},
        {"name", "destroy"},
        {"data", {}}
    };

    std::string in_topic = "sscma/v0/" + config_.client_id + "/node/in/node1";
    client.publish(in_topic, request.dump());

    auto response = client.waitMessage(out_topic, 2000);
    EXPECT_TRUE(response.has_value());

    if (response.has_value()) {
        auto msg = nlohmann::json::parse(response.value());
        EXPECT_EQ(msg["code"], 0);
    }

    // Node should be destroyed
    EXPECT_EQ(NodeFactory::instance().get("node1"), nullptr);

    server.stop();
}

TEST_F(MqttIntegrationTest, ControlRequest) {
    NodeServer server(config_);
    ASSERT_TRUE(server.start());

    for (int i = 0; i < 50 && !server.isConnected(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Create and start node
    auto* node_ptr = NodeFactory::instance().create("node1", "mqtt_test", {});
    ASSERT_NE(node_ptr, nullptr);

    std::string client_id = "client_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    MqttTestClient client(config_.host, config_.port, client_id);
    ASSERT_TRUE(client.connect());

    std::string out_topic = "sscma/v0/" + config_.client_id + "/node/out/node1";
    client.subscribe(out_topic);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send control request
    nlohmann::json request = {
        {"type", 3},
        {"name", "set_param"},
        {"data", {{"value", 42}}}
    };

    std::string in_topic = "sscma/v0/" + config_.client_id + "/node/in/node1";
    client.publish(in_topic, request.dump());

    auto response = client.waitMessage(out_topic, 2000);
    EXPECT_TRUE(response.has_value());

    if (response.has_value()) {
        auto msg = nlohmann::json::parse(response.value());
        EXPECT_EQ(msg["name"], "set_param");
        EXPECT_EQ(msg["code"], 0);
    }

    // Verify param was set
    auto* test_node = static_cast<MqttTestNode*>(node_ptr);
    EXPECT_EQ(test_node->getParam(), 42);

    server.stop();
}

TEST_F(MqttIntegrationTest, InvalidJson) {
    NodeServer server(config_);
    ASSERT_TRUE(server.start());

    for (int i = 0; i < 50 && !server.isConnected(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::string client_id = "client_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    MqttTestClient client(config_.host, config_.port, client_id);
    ASSERT_TRUE(client.connect());

    std::string out_topic = "sscma/v0/" + config_.client_id + "/node/out/node1";
    client.subscribe(out_topic);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send invalid JSON
    std::string in_topic = "sscma/v0/" + config_.client_id + "/node/in/node1";
    client.publish(in_topic, "not json");

    // Should not crash, no response expected
    auto response = client.waitMessage(out_topic, 500);
    EXPECT_FALSE(response.has_value());

    server.stop();
}

TEST_F(MqttIntegrationTest, UnknownNode) {
    NodeServer server(config_);
    ASSERT_TRUE(server.start());

    for (int i = 0; i < 50 && !server.isConnected(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::string client_id = "client_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    MqttTestClient client(config_.host, config_.port, client_id);
    ASSERT_TRUE(client.connect());

    std::string out_topic = "sscma/v0/" + config_.client_id + "/node/out/nonexistent";
    client.subscribe(out_topic);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send start request to non-existent node
    nlohmann::json request = {
        {"type", 3},
        {"name", "start"},
        {"data", {}}
    };

    std::string in_topic = "sscma/v0/" + config_.client_id + "/node/in/nonexistent";
    client.publish(in_topic, request.dump());

    auto response = client.waitMessage(out_topic, 2000);
    EXPECT_TRUE(response.has_value());

    if (response.has_value()) {
        auto msg = nlohmann::json::parse(response.value());
        EXPECT_NE(msg["code"], 0);  // Error code
    }

    server.stop();
}

TEST_F(MqttIntegrationTest, CreateWithDependencies) {
    NodeServer server(config_);
    ASSERT_TRUE(server.start());

    for (int i = 0; i < 50 && !server.isConnected(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Create camera first
    NodeFactory::instance().create("camera1", "mqtt_test", {});
    NodeFactory::instance().start("camera1");

    std::string client_id = "client_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    MqttTestClient client(config_.host, config_.port, client_id);
    ASSERT_TRUE(client.connect());

    std::string out_topic = "sscma/v0/" + config_.client_id + "/node/out/model1";
    client.subscribe(out_topic);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Create model with dependency
    nlohmann::json request = {
        {"type", 3},
        {"name", "create"},
        {"data", {
            {"type", "mqtt_test"},
            {"config", {}},
            {"dependencies", {"camera1"}}
        }}
    };

    std::string in_topic = "sscma/v0/" + config_.client_id + "/node/in/model1";
    client.publish(in_topic, request.dump());

    auto response = client.waitMessage(out_topic, 2000);
    EXPECT_TRUE(response.has_value());

    if (response.has_value()) {
        auto msg = nlohmann::json::parse(response.value());
        EXPECT_EQ(msg["code"], 0);
    }

    // Model should be auto-started (camera is ready)
    auto* model = NodeFactory::instance().get("model1");
    EXPECT_NE(model, nullptr);
    if (model) {
        EXPECT_TRUE(model->isStarted());
    }

    server.stop();
}

TEST_F(MqttIntegrationTest, EventPublishing) {
    NodeServer server(config_);
    ASSERT_TRUE(server.start());

    for (int i = 0; i < 50 && !server.isConnected(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::string client_id = "client_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    MqttTestClient client(config_.host, config_.port, client_id);
    ASSERT_TRUE(client.connect());

    std::string out_topic = "sscma/v0/" + config_.client_id + "/node/out/model1";
    client.subscribe(out_topic);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Trigger event from server
    server.event("model1", "invoke", 0, {{"boxes", nlohmann::json::array()}});

    auto response = client.waitMessage(out_topic, 1000);
    EXPECT_TRUE(response.has_value());

    if (response.has_value()) {
        auto msg = nlohmann::json::parse(response.value());
        EXPECT_EQ(msg["type"], 2);  // EVENT type
        EXPECT_EQ(msg["name"], "invoke");
    }

    server.stop();
}

TEST_F(MqttIntegrationTest, ConcurrentRequests) {
    NodeServer server(config_);
    ASSERT_TRUE(server.start());

    for (int i = 0; i < 50 && !server.isConnected(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::vector<std::thread> threads;
    std::atomic<int> success{0};

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&, i]() {
            std::string client_id = "client_" + std::to_string(i) + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
            MqttTestClient client(config_.host, config_.port, client_id);

            if (!client.connect()) return;

            std::string node_id = "node" + std::to_string(i);
            std::string out_topic = "sscma/v0/" + config_.client_id + "/node/out/" + node_id;
            client.subscribe(out_topic);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            nlohmann::json request = {
                {"type", 3},
                {"name", "create"},
                {"data", {{"type", "mqtt_test"}, {"config", {}}}}
            };

            std::string in_topic = "sscma/v0/" + config_.client_id + "/node/in/" + node_id;
            client.publish(in_topic, request.dump());

            auto response = client.waitMessage(out_topic, 2000);
            if (response.has_value()) {
                auto msg = nlohmann::json::parse(response.value());
                if (msg["code"] == 0) success++;
            }
        });
    }

    for (auto& t : threads) t.join();

    EXPECT_EQ(success.load(), 10);

    server.stop();
}
