#include "executor.h"

#include <sstream>

namespace node {

namespace {
// Generate unique executor name
std::string generateName() {
    static std::atomic<int> counter{0};
    std::ostringstream oss;
    oss << "executor_" << counter.fetch_add(1, std::memory_order_relaxed);
    return oss.str();
}
} // namespace

Executor::Executor(const std::string& name)
    : name_(name.empty() ? generateName() : name) {
    worker_ = std::thread(&Executor::workerLoop, this);
}

Executor::~Executor() {
    running_.store(false, std::memory_order_release);
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void Executor::submit(Task task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void Executor::cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<Task> empty;
    std::swap(tasks_, empty);
}

size_t Executor::pendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size();
}

void Executor::workerLoop() {
    while (running_.load(std::memory_order_acquire)) {
        Task task;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return !running_.load(std::memory_order_acquire) || !tasks_.empty();
            });

            if (!running_.load(std::memory_order_acquire)) {
                break;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        try {
            bool requeue = task();
            if (requeue) {
                submit(std::move(task));
            }
        } catch (...) {
            // Log error and continue processing
            // In production, should log the exception details
        }
    }
}

} // namespace node
