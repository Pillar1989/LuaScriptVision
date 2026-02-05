#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace node {

// Async task executor for non-blocking MQTT command processing
// Only handles control plane commands (create/start/stop/destroy)
// NOT used for data plane operations (frame processing, inference)
class Executor {
public:
    // Task function: returns true to requeue, false to complete
    using Task = std::function<bool()>;

    explicit Executor(const std::string& name = "");
    ~Executor();

    // Non-copyable, non-movable
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    Executor(Executor&&) = delete;
    Executor& operator=(Executor&&) = delete;

    // Submit task for async execution
    void submit(Task task);

    // Cancel all pending tasks
    void cancel();

    // Status queries
    size_t pendingCount() const;
    bool isRunning() const { return running_.load(std::memory_order_acquire); }
    const std::string& name() const { return name_; }

private:
    void workerLoop();

    std::string name_;
    std::queue<Task> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> running_{true};
};

} // namespace node
