#include <gtest/gtest.h>
#include <algorithm>
#include "node/error_codes.h"
#include "node/node_factory.h"

using namespace node;

// Test Node implementation for factory tests
class TestNode : public Node {
public:
    TestNode(const std::string& id, const std::string& type)
        : Node(id, type), value_(0) {}

    int onCreate(const nlohmann::json& config) override {
        (void)config;
        return MA_OK;
    }

    int onStart() override {
        return MA_OK;
    }

    int onStop() override {
        return MA_OK;
    }

    int onDestroy() override {
        return MA_OK;
    }

    int onControl(const std::string& action, const nlohmann::json& data) override {
        if (action == "set_value") {
            value_ = data.value("value", 0);
        }
        return MA_OK;
    }

    int getValue() const { return value_; }

private:
    int value_;
};

TEST(NodeFactory, Registration) {
    NodeFactory& factory = NodeFactory::instance();

    factory.registerNode("test", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    // Should be able to create registered type
    auto* node = factory.create("node1", "test", {});
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->type(), "test");

    factory.destroyAll();
}

TEST(NodeFactory, UnknownType) {
    NodeFactory& factory = NodeFactory::instance();

    auto* node = factory.create("node1", "unknown", {});
    EXPECT_EQ(node, nullptr);
}

TEST(NodeFactory, DuplicateId) {
    NodeFactory& factory = NodeFactory::instance();

    factory.registerNode("test", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    auto* node1 = factory.create("node1", "test", {});
    EXPECT_NE(node1, nullptr);

    auto* node2 = factory.create("node1", "test", {});
    EXPECT_EQ(node2, nullptr);  // Duplicate ID

    factory.destroyAll();
}

TEST(NodeFactory, Singleton) {
    NodeFactory& factory = NodeFactory::instance();

    factory.registerNode("camera", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    }, true);

    auto* cam1 = factory.create("cam1", "camera", {});
    EXPECT_NE(cam1, nullptr);

    auto* cam2 = factory.create("cam2", "camera", {});
    EXPECT_EQ(cam2, nullptr);  // Singleton already exists

    factory.destroy("cam1");

    auto* cam3 = factory.create("cam3", "camera", {});
    EXPECT_NE(cam3, nullptr);  // Can create after destroy

    factory.destroyAll();
}

TEST(NodeFactory, Dependencies) {
    NodeFactory& factory = NodeFactory::instance();

    factory.registerNode("source", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    factory.registerNode("sink", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    auto* source = factory.create("src", "source", {});
    EXPECT_NE(source, nullptr);
    factory.start("src");

    auto* sink = factory.create("snk", "sink", {}, {"src"});
    EXPECT_NE(sink, nullptr);

    // Sink should auto-start because source is ready
    EXPECT_TRUE(sink->isStarted());

    factory.destroyAll();
}

TEST(NodeFactory, MissingDependency) {
    NodeFactory& factory = NodeFactory::instance();

    factory.registerNode("sink", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    auto* sink = factory.create("snk", "sink", {}, {"nonexistent"});
    EXPECT_EQ(sink, nullptr);
}

TEST(NodeFactory, CascadingDestroy) {
    NodeFactory& factory = NodeFactory::instance();

    factory.registerNode("source", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    factory.registerNode("sink", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    auto* source = factory.create("src", "source", {});
    factory.start("src");

    factory.create("sink1", "sink", {}, {"src"});
    factory.create("sink2", "sink", {}, {"src"});

    EXPECT_EQ(factory.list().size(), 3);

    // Destroying source should destroy all sinks
    factory.destroy("src");

    EXPECT_EQ(factory.list().size(), 0);
}

TEST(NodeFactory, GetNode) {
    NodeFactory& factory = NodeFactory::instance();

    factory.registerNode("test", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    factory.create("node1", "test", {});

    auto* node = factory.get("node1");
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->id(), "node1");

    auto* missing = factory.get("nonexistent");
    EXPECT_EQ(missing, nullptr);

    factory.destroyAll();
}

TEST(NodeFactory, ListNodes) {
    NodeFactory& factory = NodeFactory::instance();

    factory.registerNode("test", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    factory.create("a", "test", {});
    factory.create("b", "test", {});
    factory.create("c", "test", {});

    auto list = factory.list();
    EXPECT_EQ(list.size(), 3);
    EXPECT_TRUE(std::find(list.begin(), list.end(), "a") != list.end());
    EXPECT_TRUE(std::find(list.begin(), list.end(), "b") != list.end());
    EXPECT_TRUE(std::find(list.begin(), list.end(), "c") != list.end());

    factory.destroyAll();
}

TEST(NodeFactory, ControlRouting) {
    NodeFactory& factory = NodeFactory::instance();

    factory.registerNode("test", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    auto* node_ptr = factory.create("node1", "test", {});
    ASSERT_NE(node_ptr, nullptr);
    factory.start("node1");

    // Cast to TestNode to access getValue
    auto* test_node = static_cast<TestNode*>(node_ptr);

    nlohmann::json data = {{"value", 42}};
    int ret = factory.control("node1", "set_value", data);
    EXPECT_EQ(ret, MA_OK);
    EXPECT_EQ(test_node->getValue(), 42);

    factory.destroyAll();
}

TEST(NodeFactory, StartStop) {
    NodeFactory& factory = NodeFactory::instance();

    factory.registerNode("test", [](const std::string& id, const std::string& type) {
        return std::make_unique<TestNode>(id, type);
    });

    // Node with no dependency will be auto-started by create()
    auto* node = factory.create("node1", "test", {});
    ASSERT_NE(node, nullptr);
    EXPECT_TRUE(node->isStarted());  // Auto-started

    // Manual stop
    EXPECT_EQ(factory.stop("node1"), MA_OK);
    EXPECT_FALSE(node->isStarted());

    // Restart
    EXPECT_EQ(factory.start("node1"), MA_OK);
    EXPECT_TRUE(node->isStarted());

    factory.destroyAll();
}

TEST(NodeFactory, DestroyNotExist) {
    NodeFactory& factory = NodeFactory::instance();

    int ret = factory.destroy("nonexistent");
    EXPECT_EQ(ret, MA_ENOENT);
}
