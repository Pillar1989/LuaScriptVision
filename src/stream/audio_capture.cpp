#include "audio_capture.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <alsa/asoundlib.h>

namespace lua_cv {

// Opaque ALSA handle structure
struct AudioCapture::AlsaHandle {
    snd_pcm_t* pcm = nullptr;
    snd_pcm_uframes_t chunk_size = 0;
    snd_pcm_uframes_t buffer_size = 0;
    int bits_per_sample = 16;
};

namespace {

// Monotonic clock (us)
uint64_t get_monotonic_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

} // namespace

AudioCapture::AudioCapture(const Config& config)
    : config_(config) {
}

AudioCapture::~AudioCapture() {
    if (running_.load()) {
        std::cerr << "[AudioCapture] Warning: stop() not called before destruction" << std::endl;
        stop();
    }
}

bool AudioCapture::start() {
    if (running_.load()) {
        std::cerr << "[AudioCapture] Already running" << std::endl;
        return false;
    }

    // Initialize ALSA
    if (!init_alsa()) {
        return false;
    }

    // Set initial volume
    set_volume_hw(config_.volume);

    // Start capture thread
    running_.store(true);
    capture_thread_ = std::thread(&AudioCapture::capture_thread_func, this);

    std::cout << "[AudioCapture] Started: device=" << config_.device
              << ", rate=" << config_.sample_rate
              << ", channels=" << config_.channels
              << ", volume=" << config_.volume << std::endl;

    return true;
}

void AudioCapture::stop() {
    if (!running_.load()) {
        return;
    }

    std::cout << "[AudioCapture] Stopping..." << std::endl;
    running_.store(false);

    // Notify queue to unblock get_frame()
    queue_cv_.notify_all();

    // Wait for capture thread to finish
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }

    // Shutdown ALSA
    shutdown_alsa();

    // Clear queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!frame_queue_.empty()) {
            frame_queue_.pop();
        }
    }

    std::cout << "[AudioCapture] Stopped" << std::endl;
}

bool AudioCapture::get_frame(AudioFrame* frame, int timeout_ms) {
    if (!frame) {
        return false;
    }

    std::unique_lock<std::mutex> lock(queue_mutex_);

    // Wait for frame with timeout
    bool has_frame = queue_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
        [this] { return !frame_queue_.empty() || !running_.load(); });

    if (!has_frame || frame_queue_.empty() || !running_.load()) {
        return false;
    }

    // Move frame to output
    *frame = std::move(frame_queue_.front());
    frame_queue_.pop();

    return true;
}

void AudioCapture::set_volume(int volume) {
    config_.volume = std::clamp(volume, 0, 100);
    set_volume_hw(config_.volume);
}

bool AudioCapture::init_alsa() {
    alsa_handle_ = new AlsaHandle();
    snd_pcm_hw_params_t* params;
    int err;

    // Open PCM device
    err = snd_pcm_open(&alsa_handle_->pcm, config_.device.c_str(),
                       SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        std::cerr << "[AudioCapture] snd_pcm_open failed: " << snd_strerror(err) << std::endl;
        delete alsa_handle_;
        alsa_handle_ = nullptr;
        return false;
    }

    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(alsa_handle_->pcm, params);

    // Set access mode
    err = snd_pcm_hw_params_set_access(alsa_handle_->pcm, params,
                                        SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
        std::cerr << "[AudioCapture] set_access failed: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_->pcm);
        delete alsa_handle_;
        alsa_handle_ = nullptr;
        return false;
    }

    // Set format (16-bit little-endian)
    err = snd_pcm_hw_params_set_format(alsa_handle_->pcm, params,
                                        SND_PCM_FORMAT_S16_LE);
    if (err < 0) {
        std::cerr << "[AudioCapture] set_format failed: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_->pcm);
        delete alsa_handle_;
        alsa_handle_ = nullptr;
        return false;
    }

    // Set channels
    err = snd_pcm_hw_params_set_channels(alsa_handle_->pcm, params, config_.channels);
    if (err < 0) {
        std::cerr << "[AudioCapture] set_channels failed: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_->pcm);
        delete alsa_handle_;
        alsa_handle_ = nullptr;
        return false;
    }

    // Set sample rate
    unsigned int rate = config_.sample_rate;
    err = snd_pcm_hw_params_set_rate_near(alsa_handle_->pcm, params, &rate, 0);
    if (err < 0) {
        std::cerr << "[AudioCapture] set_rate failed: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_->pcm);
        delete alsa_handle_;
        alsa_handle_ = nullptr;
        return false;
    }

    // Configure buffer/period time
    unsigned int buffer_time;
    err = snd_pcm_hw_params_get_buffer_time_max(params, &buffer_time, 0);
    if (err < 0) {
        std::cerr << "[AudioCapture] get_buffer_time_max failed: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_->pcm);
        delete alsa_handle_;
        alsa_handle_ = nullptr;
        return false;
    }
    if (buffer_time > 100000) buffer_time = 100000;

    unsigned int period_time = buffer_time / 4;
    err = snd_pcm_hw_params_set_period_time_near(alsa_handle_->pcm, params, &period_time, 0);
    if (err < 0) {
        std::cerr << "[AudioCapture] set_period_time failed: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_->pcm);
        delete alsa_handle_;
        alsa_handle_ = nullptr;
        return false;
    }

    err = snd_pcm_hw_params_set_buffer_time_near(alsa_handle_->pcm, params, &buffer_time, 0);
    if (err < 0) {
        std::cerr << "[AudioCapture] set_buffer_time failed: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_->pcm);
        delete alsa_handle_;
        alsa_handle_ = nullptr;
        return false;
    }

    // Apply parameters
    err = snd_pcm_hw_params(alsa_handle_->pcm, params);
    if (err < 0) {
        std::cerr << "[AudioCapture] snd_pcm_hw_params failed: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_->pcm);
        delete alsa_handle_;
        alsa_handle_ = nullptr;
        return false;
    }

    // Prepare device
    err = snd_pcm_prepare(alsa_handle_->pcm);
    if (err < 0) {
        std::cerr << "[AudioCapture] snd_pcm_prepare failed: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_->pcm);
        delete alsa_handle_;
        alsa_handle_ = nullptr;
        return false;
    }

    // Get buffer sizes for frame allocation
    snd_pcm_hw_params_get_period_size(params, &alsa_handle_->chunk_size, 0);
    snd_pcm_hw_params_get_buffer_size(params, &alsa_handle_->buffer_size);

    std::cout << "[AudioCapture] ALSA initialized: chunk_size=" << alsa_handle_->chunk_size
              << ", buffer_size=" << alsa_handle_->buffer_size << std::endl;

    return true;
}

void AudioCapture::shutdown_alsa() {
    if (alsa_handle_) {
        if (alsa_handle_->pcm) {
            snd_pcm_drain(alsa_handle_->pcm);
            snd_pcm_close(alsa_handle_->pcm);
        }
        delete alsa_handle_;
        alsa_handle_ = nullptr;
    }
}

bool AudioCapture::set_volume_hw(int volume) {
    snd_mixer_t* mixer = nullptr;
    snd_mixer_elem_t* elem = nullptr;
    int err;

    // Map 0-100 to hardware range 0-24
    int hw_volume = volume * 24 / 100;

    // Open mixer
    err = snd_mixer_open(&mixer, 0);
    if (err < 0) {
        std::cerr << "[AudioCapture] Failed to open mixer: " << snd_strerror(err) << std::endl;
        return false;
    }

    // Attach to audio device
    err = snd_mixer_attach(mixer, config_.device.c_str());
    if (err < 0) {
        std::cerr << "[AudioCapture] Failed to attach mixer: " << snd_strerror(err) << std::endl;
        snd_mixer_close(mixer);
        return false;
    }

    // Register simple mixer element interface
    err = snd_mixer_selem_register(mixer, nullptr, nullptr);
    if (err < 0) {
        std::cerr << "[AudioCapture] Failed to register mixer: " << snd_strerror(err) << std::endl;
        snd_mixer_close(mixer);
        return false;
    }

    // Load mixer elements
    err = snd_mixer_load(mixer);
    if (err < 0) {
        std::cerr << "[AudioCapture] Failed to load mixer: " << snd_strerror(err) << std::endl;
        snd_mixer_close(mixer);
        return false;
    }

    // Find "ADC" capture volume element
    for (elem = snd_mixer_first_elem(mixer); elem; elem = snd_mixer_elem_next(elem)) {
        // Only process simple mixer elements
        if (snd_mixer_elem_get_type(elem) != SND_MIXER_ELEM_SIMPLE) {
            continue;
        }

        const char* name = snd_mixer_selem_get_name(elem);
        if (name && strcmp(name, "ADC") == 0) {
            if (snd_mixer_selem_has_capture_volume(elem)) {
                break;
            }
        }
    }

    if (!elem) {
        std::cerr << "[AudioCapture] Failed to find ADC mixer element" << std::endl;
        snd_mixer_close(mixer);
        return false;
    }

    // Check if element has capture volume
    if (!snd_mixer_selem_has_capture_volume(elem)) {
        std::cerr << "[AudioCapture] ADC element does not have capture volume" << std::endl;
        snd_mixer_close(mixer);
        return false;
    }

    // Set capture volume for both channels (stereo)
    err = snd_mixer_selem_set_capture_volume(elem, SND_MIXER_SCHN_FRONT_LEFT, hw_volume);
    if (err < 0) {
        std::cerr << "[AudioCapture] Failed to set left volume: " << snd_strerror(err) << std::endl;
        snd_mixer_close(mixer);
        return false;
    }

    err = snd_mixer_selem_set_capture_volume(elem, SND_MIXER_SCHN_FRONT_RIGHT, hw_volume);
    if (err < 0) {
        std::cerr << "[AudioCapture] Failed to set right volume: " << snd_strerror(err) << std::endl;
        snd_mixer_close(mixer);
        return false;
    }

    std::cout << "[AudioCapture] Volume set to " << volume
              << " (hw=" << hw_volume << ")" << std::endl;

    snd_mixer_close(mixer);
    return true;
}

void AudioCapture::capture_thread_func() {
    if (!alsa_handle_ || !alsa_handle_->pcm) {
        std::cerr << "[AudioCapture] Invalid ALSA handle in capture thread" << std::endl;
        return;
    }

    // Allocate buffer (chunk_size * channels * bits_per_sample / 8)
    size_t buffer_bytes = alsa_handle_->chunk_size * config_.channels *
                          (alsa_handle_->bits_per_sample / 8);
    std::vector<uint16_t> buffer(buffer_bytes / 2 + 1);  // uint16_t = 2 bytes

    std::cout << "[AudioCapture] Capture thread started" << std::endl;

    while (running_.load()) {
        // Read from ALSA
        snd_pcm_sframes_t pcm_return = snd_pcm_readi(alsa_handle_->pcm, buffer.data(),
                                                      alsa_handle_->chunk_size);

        if (pcm_return == -EPIPE) {
            // XRUN: recoverable error
            std::cerr << "[AudioCapture] XRUN occurred" << std::endl;
            if (snd_pcm_prepare(alsa_handle_->pcm) < 0) {
                std::cerr << "[AudioCapture] XRUN recovery failed, exiting" << std::endl;
                break;
            }
            continue;
        } else if (pcm_return == -ESTRPIPE) {
            // Suspend: fatal (fail-fast)
            std::cerr << "[AudioCapture] ALSA suspended (fatal)" << std::endl;
            break;
        } else if (pcm_return < 0) {
            // Other errors: fatal
            std::cerr << "[AudioCapture] ALSA read error: " << snd_strerror(pcm_return)
                      << " (fatal)" << std::endl;
            break;
        }

        // Create audio frame
        AudioFrame frame;
        frame.data.resize(buffer_bytes);
        std::memcpy(frame.data.data(), buffer.data(), buffer_bytes);
        frame.samples = pcm_return * config_.channels;
        frame.pts = get_monotonic_time_us();

        // Push to queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (frame_queue_.size() < config_.frame_queue_size) {
                frame_queue_.push(std::move(frame));
            } else {
                std::cerr << "[AudioCapture] Frame queue full, dropping frame" << std::endl;
            }
        }
        queue_cv_.notify_one();
    }

    std::cout << "[AudioCapture] Capture thread exited" << std::endl;
}

} // namespace lua_cv
