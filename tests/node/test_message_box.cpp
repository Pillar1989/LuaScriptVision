#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>

#include "node/message_box.h"
#include "node/pipeline_context.h"

#include <opencv2/opencv.hpp>

using namespace node;

namespace {
lua_cv::Frame make_test_frame(int width, int height) {
    cv::Mat mat = cv::Mat::zeros(height, width, CV_8UC3);
    return lua_cv::Frame(std::move(mat));
}
}  // namespace

// ============================================================================
// MessageBox Basic Tests
// ============================================================================

TEST(MessageBox, BasicPostFetch) {
    MessageBox<int> mbox(4);

    int* val = new int(42);
    EXPECT_TRUE(mbox.post(val, 0));
    EXPECT_EQ(mbox.count(), 1);

    int* result;
    EXPECT_TRUE(mbox.fetch(&result, 0));
    EXPECT_EQ(*result, 42);
    EXPECT_TRUE(mbox.isEmpty());

    delete result;
}

TEST(MessageBox, QueueFull) {
    MessageBox<int> mbox(2);

    int* v1 = new int(1);
    int* v2 = new int(2);
    int* v3 = new int(3);

    EXPECT_TRUE(mbox.post(v1, 0));
    EXPECT_TRUE(mbox.post(v2, 0));
    EXPECT_TRUE(mbox.isFull());

    // Should fail immediately (non-blocking)
    EXPECT_FALSE(mbox.post(v3, 0));

    delete v3;

    // Cleanup
    int* tmp;
    mbox.fetch(&tmp, 0); delete tmp;
    mbox.fetch(&tmp, 0); delete tmp;
}

TEST(MessageBox, BlockingFetch) {
    MessageBox<int> mbox(4);

    std::thread producer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        mbox.post(new int(42), 0);
    });

    auto start = std::chrono::steady_clock::now();

    int* result;
    EXPECT_TRUE(mbox.fetch(&result, 1000));
    EXPECT_EQ(*result, 42);

    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_GE(elapsed, std::chrono::milliseconds(90));
    EXPECT_LE(elapsed, std::chrono::milliseconds(300));

    delete result;
    producer.join();
}

TEST(MessageBox, FetchTimeout) {
    MessageBox<int> mbox(4);

    auto start = std::chrono::steady_clock::now();

    int* result;
    EXPECT_FALSE(mbox.fetch(&result, 100));

    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_GE(elapsed, std::chrono::milliseconds(90));
    EXPECT_LE(elapsed, std::chrono::milliseconds(200));
}

TEST(MessageBox, FifoOrder) {
    MessageBox<int> mbox(4);

    for (int i = 0; i < 4; i++) {
        mbox.post(new int(i), 0);
    }

    for (int i = 0; i < 4; i++) {
        int* result;
        EXPECT_TRUE(mbox.fetch(&result, 0));
        EXPECT_EQ(*result, i);
        delete result;
    }
}

TEST(MessageBox, ProducerConsumer) {
    MessageBox<int> mbox(4);
    std::atomic<int> sum{0};
    std::atomic<bool> done{false};

    std::thread producer([&]() {
        for (int i = 1; i <= 100; i++) {
            while (!mbox.post(new int(i), 10)) {
                // Retry
            }
        }
        done = true;
    });

    std::thread consumer([&]() {
        while (!done || !mbox.isEmpty()) {
            int* val;
            if (mbox.fetch(&val, 50)) {
                sum += *val;
                delete val;
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(sum.load(), 5050);  // 1+2+...+100
}

TEST(MessageBox, MultipleConsumers) {
    MessageBox<int> mbox(8);
    std::atomic<int> consumed{0};
    std::atomic<bool> producer_done{false};

    // Producer
    std::thread producer([&]() {
        for (int i = 0; i < 100; i++) {
            while (!mbox.post(new int(i), 10)) {}
        }
        producer_done = true;
    });

    // Multiple consumers
    auto consumer_func = [&]() {
        while (consumed.load() < 100) {
            int* val;
            if (mbox.fetch(&val, 50)) {
                consumed++;
                delete val;
            }
            if (producer_done && mbox.isEmpty()) break;
        }
    };

    std::thread c1(consumer_func);
    std::thread c2(consumer_func);
    std::thread c3(consumer_func);
    std::thread c4(consumer_func);

    producer.join();
    c1.join();
    c2.join();
    c3.join();
    c4.join();

    EXPECT_EQ(consumed.load(), 100);
}

TEST(MessageBox, Clear) {
    MessageBox<int> mbox(4);

    mbox.post(new int(1), 0);
    mbox.post(new int(2), 0);
    mbox.post(new int(3), 0);

    EXPECT_EQ(mbox.count(), 3);

    // Note: clear() will delete the items via deleter or delete
    mbox.clear();

    EXPECT_TRUE(mbox.isEmpty());
}

TEST(MessageBox, Interrupt) {
    MessageBox<int> mbox(4);

    std::atomic<bool> fetch_returned{false};

    std::thread fetcher([&]() {
        int* result;
        mbox.fetch(&result, UINT32_MAX);  // Block forever
        fetch_returned = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(fetch_returned.load());

    mbox.interrupt();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(fetch_returned.load());

    fetcher.join();
}

TEST(MessageBox, Capacity) {
    MessageBox<int> mbox(10);
    EXPECT_EQ(mbox.capacity(), 10);
}

TEST(MessageBox, BlockingPost) {
    MessageBox<int> mbox(2);

    mbox.post(new int(1), 0);
    mbox.post(new int(2), 0);

    std::atomic<bool> post_returned{false};

    std::thread poster([&]() {
        mbox.post(new int(3), UINT32_MAX);  // Block until space available
        post_returned = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(post_returned.load());

    // Free up space
    int* tmp;
    mbox.fetch(&tmp, 0);
    delete tmp;

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(post_returned.load());

    poster.join();

    // Cleanup
    mbox.fetch(&tmp, 0); delete tmp;
    mbox.fetch(&tmp, 0); delete tmp;
}

TEST(MessageBox, InterruptPost) {
    MessageBox<int> mbox(1);

    mbox.post(new int(1), 0);  // Fill the queue

    std::atomic<bool> post_returned{false};

    std::thread poster([&]() {
        bool result = mbox.post(new int(2), UINT32_MAX);  // Block forever
        (void)result;
        post_returned = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(post_returned.load());

    mbox.interrupt();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(post_returned.load());

    poster.join();

    // Cleanup - need to handle the item that was blocked
    mbox.clear();
}

TEST(MessageBox, Reset) {
    MessageBox<int> mbox(4);

    mbox.interrupt();

    // After interrupt, fetch should fail immediately
    int* result;
    EXPECT_FALSE(mbox.fetch(&result, 100));

    mbox.reset();

    // After reset, fetch should work normally (timeout)
    auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(mbox.fetch(&result, 100));
    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_GE(elapsed, std::chrono::milliseconds(90));
}

// ============================================================================
// SharedFrame + PipelineContext Tests (Reference Count & Zero-Copy)
// ============================================================================

TEST(SharedFrame, ReferenceCount) {
    auto* sf = new SharedFrame(make_test_frame(100, 100));
    EXPECT_EQ(sf->ref_count(), 1);

    sf->ref(2);
    EXPECT_EQ(sf->ref_count(), 3);

    sf->release();
    EXPECT_EQ(sf->ref_count(), 2);

    sf->release();
    EXPECT_EQ(sf->ref_count(), 1);

    sf->release();
}

TEST(SharedFrame, MultiSubscriberBroadcast) {
    MessageBox<PipelineContext> mbox1(4);
    MessageBox<PipelineContext> mbox2(4);

    auto* sf = new SharedFrame(make_test_frame(128, 128));

    sf->ref();
    auto* ctx1 = new PipelineContext{sf, nlohmann::json::object(), 1};
    sf->ref();
    auto* ctx2 = new PipelineContext{sf, nlohmann::json::object(), 1};

    EXPECT_TRUE(mbox1.post(ctx1, 0));
    EXPECT_TRUE(mbox2.post(ctx2, 0));

    PipelineContext* recv1 = nullptr;
    PipelineContext* recv2 = nullptr;
    EXPECT_TRUE(mbox1.fetch(&recv1, 0));
    EXPECT_TRUE(mbox2.fetch(&recv2, 0));

    EXPECT_EQ(recv1->frame, sf);
    EXPECT_EQ(recv2->frame, sf);

    delete recv1;
    EXPECT_EQ(sf->ref_count(), 2);

    delete recv2;
    EXPECT_EQ(sf->ref_count(), 1);

    sf->release();
}

TEST(PipelineContext, BasicPass) {
    MessageBox<PipelineContext> mbox(4);

    auto* sf = new SharedFrame(make_test_frame(640, 640));

    nlohmann::json boxes = nlohmann::json::array();
    boxes.push_back({{"x", 100}, {"y", 200}, {"w", 50}, {"h", 80}, {"score", 0.95}});
    boxes.push_back({{"x", 300}, {"y", 150}, {"w", 60}, {"h", 90}, {"score", 0.87}});

    sf->ref();
    auto* ctx = new PipelineContext{sf, boxes, 12345};

    EXPECT_TRUE(mbox.post(ctx, 0));

    PipelineContext* recv = nullptr;
    EXPECT_TRUE(mbox.fetch(&recv, 0));

    EXPECT_EQ(recv->frame, sf);
    EXPECT_EQ(recv->frame_id, 12345u);
    EXPECT_TRUE(recv->upstream_result.is_array());
    EXPECT_EQ(recv->upstream_result.size(), 2u);

    delete recv;
    EXPECT_EQ(sf->ref_count(), 1);
    sf->release();
}

TEST(PipelineContext, ChainedPass) {
    MessageBox<PipelineContext> mbox_ab(4);
    MessageBox<PipelineContext> mbox_bc(4);

    auto* sf = new SharedFrame(make_test_frame(640, 640));

    nlohmann::json detect_result = nlohmann::json::array();
    detect_result.push_back({{"x", 100}, {"y", 200}, {"w", 50}, {"h", 80}});

    sf->ref();
    auto* ctx_ab = new PipelineContext{sf, detect_result, 1};
    EXPECT_TRUE(mbox_ab.post(ctx_ab, 0));

    PipelineContext* recv_b = nullptr;
    EXPECT_TRUE(mbox_ab.fetch(&recv_b, 0));
    EXPECT_EQ(recv_b->upstream_result.size(), 1u);

    nlohmann::json classify_result = nlohmann::json::array();
    classify_result.push_back({{"class_id", 0}, {"label", "person"}, {"confidence", 0.98}});

    recv_b->frame->ref();
    auto* ctx_bc = new PipelineContext{recv_b->frame, classify_result, 1};
    EXPECT_TRUE(mbox_bc.post(ctx_bc, 0));

    delete recv_b;
    EXPECT_EQ(sf->ref_count(), 2);

    PipelineContext* recv_c = nullptr;
    EXPECT_TRUE(mbox_bc.fetch(&recv_c, 0));
    EXPECT_EQ(recv_c->upstream_result[0]["class_id"], 0);

    delete recv_c;
    EXPECT_EQ(sf->ref_count(), 1);

    sf->release();
}
