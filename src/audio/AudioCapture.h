#pragma once

#include <vector>
#include <atomic>
#include <mutex>
#include <functional>
#include <memory>

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();

    // Initialize audio capture (loopback mode for system audio)
    bool initialize(unsigned int sampleRate = 44100, unsigned int channels = 2);

    // Start/stop capture
    bool start();
    void stop();

    // Check if currently capturing
    bool isCapturing() const { return m_isCapturing; }

    // Get latest audio data (thread-safe)
    // Returns false if no new data available
    bool getLatestAudioData(std::vector<float>& outBuffer);

    // Set callback for when new audio data is available
    void setAudioCallback(std::function<void(const float* data, unsigned int frameCount)> callback);

    // Get audio properties
    unsigned int getSampleRate() const { return m_sampleRate; }
    unsigned int getChannels() const { return m_channels; }

private:
    // PIMPL idiom to hide miniaudio types
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    unsigned int m_sampleRate;
    unsigned int m_channels;
    std::atomic<bool> m_isCapturing;
    std::atomic<bool> m_hasInitialized;

    // Simple fixed buffer for latest audio chunk
    std::vector<float> m_audioBuffer;
    std::mutex m_bufferMutex;

    // Optional callback
    std::function<void(const float* data, unsigned int frameCount)> m_audioCallback;

    // Instance callback (called from miniaudio callback in Impl)
    void onAudioData(const float* pInput, unsigned int frameCount);
};
