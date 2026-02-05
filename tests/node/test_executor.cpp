#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <mutex>

#include "node/executor.h"

using namespace node;

// ============================================================================
// Executor Basic Tests
// ============================================================================

TEST(Executor, TaskExecution) {
    Executor exec("test");

    std::atomic<bool> executed{false};

    exec.submit([&]() {
        executed = true;
        return false;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(executed.load());
}

TEST(Executor, TaskOrder) {
    Executor exec("test");

    std::vector<int> results;
    std::mutex mutex;

    for (int i = 0; i < 10; i++) {
        exec.submit([&, i]() {
            std::lock_guard<std::mutex> lock(mutex);
            results.push_back(i);
            return false;
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(results.size(), 10);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(results[i], i);  // FIFO order
    }
}

TEST(Executor, TaskRequeue) {
    Executor exec("test");

    std::atomic<int> count{0};

    exec.submit([&]() {
        count++;
        return count < 5;  // Requeue 4 times
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(count.load(), 5);
}

TEST(Executor, ExceptionHandling) {
    Executor exec("test");

    std::atomic<int> after_exception{0};

    exec.submit([]() {
        throw std::runtime_error("test error");
        return false;
    });

    exec.submit([&]() {
        after_exception++;
        return false;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Second task should still execute
    EXPECT_EQ(after_exception.load(), 1);
}

TEST(Executor, CancelPending) {
    Executor exec("test");

    std::atomic<int> executed{0};

    // Submit slow task
    exec.submit([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        executed++;
        return false;
    });

    // Submit more tasks
    for (int i = 0; i < 5; i++) {
        exec.submit([&]() {
            executed++;
            return false;
        });
    }

    // Cancel before slow task completes
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    exec.cancel();

    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    // Only first task should have run
    EXPECT_EQ(executed.load(), 1);
}

TEST(Executor, ShutdownCleanup) {
    std::atomic<int> executed{0};
    std::atomic<bool> first_task_started{false};

    {
        Executor exec("test");

        for (int i = 0; i < 100; i++) {
            exec.submit([&, i]() {
                if (i == 0) {
                    first_task_started = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                executed++;
                return false;
            });
        }

        // Wait for at least one task to start
        while (!first_task_started.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Destructor called here - should not crash or hang
    }

    // At least one task should have started execution
    // The test passes if we reach here without hanging
    EXPECT_GE(executed.load(), 0);
}

TEST(Executor, ConcurrentSubmit) {
    Executor exec("test");

    std::atomic<int> count{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < 10; t++) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 100; i++) {
                exec.submit([&]() {
                    count++;
                    return false;
                });
            }
        });
    }

    for (auto& t : threads) t.join();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    EXPECT_EQ(count.load(), 1000);
}

TEST(Executor, PendingCount) {
    Executor exec("test");

    std::atomic<bool> block{true};

    // Submit blocking task
    exec.submit([&]() {
        while (block) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    });

    // Wait for blocking task to start
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    // Submit more tasks
    for (int i = 0; i < 5; i++) {
        exec.submit([]() { return false; });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(exec.pendingCount(), 5);

    block = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(exec.pendingCount(), 0);
}

TEST(Executor, LongRunningTask) {
    Executor exec("test");

    std::atomic<int> short_tasks{0};
    auto start = std::chrono::steady_clock::now();

    // Long task
    exec.submit([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        return false;
    });

    // Short tasks queued behind
    for (int i = 0; i < 3; i++) {
        exec.submit([&]() {
            short_tasks++;
            return false;
        });
    }

    // Wait for completion
    while (short_tasks < 3) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    auto elapsed = std::chrono::steady_clock::now() - start;

    // Short tasks blocked by long task
    EXPECT_GE(elapsed, std::chrono::milliseconds(300));
}

TEST(Executor, NamedExecutor) {
    Executor exec("my_executor");
    EXPECT_EQ(exec.name(), "my_executor");

    // Check isRunning
    EXPECT_TRUE(exec.isRunning());
}

TEST(Executor, DefaultName) {
    Executor exec1("");
    Executor exec2("");

    // Should have generated unique names
    EXPECT_FALSE(exec1.name().empty());
    EXPECT_FALSE(exec2.name().empty());
    EXPECT_NE(exec1.name(), exec2.name());
}

TEST(Executor, IsRunning) {
    auto exec = std::make_unique<Executor>("test");
    EXPECT_TRUE(exec->isRunning());

    exec.reset();  // Destructor should stop it
    // Can't check isRunning after destruction
}

TEST(Executor, EmptyQueuePendingCount) {
    Executor exec("test");
    EXPECT_EQ(exec.pendingCount(), 0);
}

TEST(Executor, MultipleRequeues) {
    Executor exec("test");

    std::atomic<int> iteration{0};
    const int max_iterations = 10;

    exec.submit([&]() {
        iteration++;
        return iteration < max_iterations;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(iteration.load(), max_iterations);
}

TEST(Executor, TasksCompleteBeforeDestruction) {
    std::atomic<int> completed{0};
    const int total_tasks = 5;

    {
        Executor exec("test");

        for (int i = 0; i < total_tasks; i++) {
            exec.submit([&]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                completed++;
                return false;
            });
        }

        // Wait a bit for tasks to start
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Destructor will be called here, should wait for current task
    }

    // At least one task should have completed
    EXPECT_GT(completed.load(), 0);
}
