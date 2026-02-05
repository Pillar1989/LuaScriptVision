#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include "node/error_codes.h"
#include "node/node.h"

using namespace node;

// Test Node implementation
class TestNode : public Node {
public:
    TestNode(const std::string& id, const std::string& type)
        : Node(id, type) {}

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
        (void)action;
        (void)data;
        return MA_OK;
    }
};

TEST(Node, BasicConstruction) {
    TestNode node("test1", "test");
    EXPECT_EQ(node.id(), "test1");
    EXPECT_EQ(node.type(), "test");
    EXPECT_FALSE(node.isCreated());
    EXPECT_FALSE(node.isStarted());
}

TEST(Node, Create) {
    TestNode node("test1", "test");
    EXPECT_EQ(node.create({}), MA_OK);
    EXPECT_TRUE(node.isCreated());
    EXPECT_FALSE(node.isStarted());
}

TEST(Node, DuplicateCreate) {
    TestNode node("test1", "test");
    EXPECT_EQ(node.create({}), MA_OK);
    EXPECT_EQ(node.create({}), MA_EEXIST);
}

TEST(Node, StartWithoutCreate) {
    TestNode node("test1", "test");
    EXPECT_EQ(node.start(), MA_EINVAL);
    EXPECT_FALSE(node.isStarted());
}

TEST(Node, NormalLifecycle) {
    TestNode node("test1", "test");

    EXPECT_EQ(node.create({}), MA_OK);
    EXPECT_EQ(node.start(), MA_OK);
    EXPECT_TRUE(node.isStarted());

    EXPECT_EQ(node.stop(), MA_OK);
    EXPECT_FALSE(node.isStarted());

    EXPECT_EQ(node.destroy(), MA_OK);
    EXPECT_FALSE(node.isCreated());
}

TEST(Node, DependencyNotReady) {
    TestNode nodeA("a", "test");
    TestNode nodeB("b", "test");

    nodeA.create({});
    nodeB.create({});
    nodeB.addDependency(&nodeA);

    // B cannot start because A not started
    EXPECT_EQ(nodeB.start(), MA_EAGAIN);

    // Start A, then B can start
    nodeA.start();
    EXPECT_EQ(nodeB.start(), MA_OK);
}

TEST(Node, CascadingStop) {
    TestNode nodeA("a", "test");
    TestNode nodeB("b", "test");

    nodeA.create({});
    nodeB.create({});
    nodeB.addDependency(&nodeA);

    nodeA.start();
    nodeB.start();

    EXPECT_TRUE(nodeA.isStarted());
    EXPECT_TRUE(nodeB.isStarted());

    // Stopping A should stop B first
    nodeA.stop();
    EXPECT_FALSE(nodeA.isStarted());
    EXPECT_FALSE(nodeB.isStarted());
}

TEST(Node, ThreadSafety) {
    TestNode node("test1", "test");
    node.create({});

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    // Multiple threads try to start - all should succeed (idempotent)
    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&]() {
            int ret = node.start();
            if (ret == MA_OK) {
                success_count++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All calls should succeed (idempotent start)
    EXPECT_EQ(success_count.load(), 10);
    EXPECT_TRUE(node.isStarted());
}
