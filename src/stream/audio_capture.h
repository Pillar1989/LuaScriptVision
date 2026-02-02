#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace lua_cv {

/**
 * Audio capture using ALSA.
 *
 * Thread model: Push model (internal capture thread + frame queue).
 * - Capture thread continuously reads from ALSA and pushes frames to internal queue.
 * - get_frame() pops from queue with timeout.
 */
class AudioCapture {
public:
    struct AudioFrame {
        std::vector<uint8_t> data;    // PCM audio data (interleaved)
        uint64_t pts;                  // Monotonic timestamp (microseconds)
        uint32_t samples;              // Number of samples in this frame
    };

    struct Config {
        std::string device = "hw:0";
        uint32_t sample_rate = 16000;
        uint32_t channels = 1;
        uint32_t bit_width = 16;
        uint32_t volume = 80;              // 0-100 (mapped to 0-24 hardware range)
        size_t frame_queue_size = 32;      // Max frames in queue (~640ms at 16kHz)
    };

    explicit AudioCapture(const Config& config);
    ~AudioCapture();

    AudioCapture(const AudioCapture&) = delete;
    AudioCapture& operator=(const AudioCapture&) = delete;
    AudioCapture(AudioCapture&&) = delete;
    AudioCapture& operator=(AudioCapture&&) = delete;

    /**
     * Start audio capture.
     * Opens ALSA device, spawns capture thread.
     * Returns false on error.
     */
    bool start();

    /**
     * Stop audio capture.
     * Stops capture thread, closes ALSA device.
     */
    void stop();

    bool is_running() const { return running_.load(); }

    /**
     * Get audio frame from internal queue.
     *
     * @param frame Output audio frame
     * @param timeout_ms Timeout in milliseconds (default: 100ms)
     * @return true if frame retrieved, false on timeout or error
     */
    bool get_frame(AudioFrame* frame, int timeout_ms = 100);

    /**
     * Set capture volume (0-100).
     * Maps to hardware range 0-24 via amixer.
     */
    void set_volume(int volume);

    /**
     * Get current volume (0-100).
     */
    int get_volume() const { return config_.volume; }

    const Config& config() const { return config_; }

private:
    // ALSA handle (opaque pointer to avoid including alsa headers in public header)
    struct AlsaHandle;
    AlsaHandle* alsa_handle_ = nullptr;

    Config config_;
    std::atomic<bool> running_{false};
    std::thread capture_thread_;

    // Frame queue for push model
    std::queue<AudioFrame> frame_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    void capture_thread_func();
    bool init_alsa();
    void shutdown_alsa();
    bool set_volume_hw(int volume);
};

} // namespace lua_cv
