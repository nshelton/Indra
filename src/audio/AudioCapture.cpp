#define MINIAUDIO_IMPLEMENTATION
#include "AudioCapture.h"
#include <glog/logging.h>
#include <algorithm>

AudioCapture::AudioCapture()
    : m_sampleRate(44100)
    , m_channels(2)
    , m_isCapturing(false)
    , m_bufferSize(44100 * 2) // 1 second of stereo audio at 44.1kHz
    , m_writePos(0)
    , m_hasNewData(false)
{
    m_audioBuffer.resize(m_bufferSize, 0.0f);
}

AudioCapture::~AudioCapture()
{
    stop();
    ma_device_uninit(&m_device);
}

bool AudioCapture::initialize(unsigned int sampleRate, unsigned int channels)
{
    m_sampleRate = sampleRate;
    m_channels = channels;
    m_bufferSize = sampleRate * channels; // 1 second buffer
    m_audioBuffer.resize(m_bufferSize, 0.0f);

    // Configure device for loopback capture
    m_deviceConfig = ma_device_config_init(ma_device_type_loopback);
    m_deviceConfig.capture.format = ma_format_f32; // 32-bit float
    m_deviceConfig.capture.channels = channels;
    m_deviceConfig.sampleRate = sampleRate;
    m_deviceConfig.dataCallback = &AudioCapture::dataCallback;
    m_deviceConfig.pUserData = this;

    // Initialize the device
    ma_result result = ma_device_init(NULL, &m_deviceConfig, &m_device);
    if (result != MA_SUCCESS) {
        LOG(ERROR) << "Failed to initialize audio capture device: " << result;
        return false;
    }

    LOG(INFO) << "Audio capture initialized: " << sampleRate << "Hz, " << channels << " channels";
    return true;
}

bool AudioCapture::start()
{
    if (m_isCapturing) {
        LOG(WARNING) << "Audio capture already running";
        return false;
    }

    ma_result result = ma_device_start(&m_device);
    if (result != MA_SUCCESS) {
        LOG(ERROR) << "Failed to start audio capture: " << result;
        return false;
    }

    m_isCapturing = true;
    LOG(INFO) << "Audio capture started";
    return true;
}

void AudioCapture::stop()
{
    if (!m_isCapturing) {
        return;
    }

    ma_device_stop(&m_device);
    m_isCapturing = false;
    LOG(INFO) << "Audio capture stopped";
}

bool AudioCapture::getLatestAudioData(std::vector<float>& outBuffer)
{
    if (!m_hasNewData) {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_bufferMutex);
    outBuffer = m_audioBuffer;
    m_hasNewData = false;
    return true;
}

void AudioCapture::setAudioCallback(std::function<void(const float* data, unsigned int frameCount)> callback)
{
    m_audioCallback = callback;
}

void AudioCapture::dataCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    AudioCapture* pCapture = static_cast<AudioCapture*>(pDevice->pUserData);
    if (pCapture && pInput) {
        pCapture->onAudioData(static_cast<const float*>(pInput), frameCount);
    }

    // For loopback mode, we don't output anything
    (void)pOutput;
}

void AudioCapture::onAudioData(const float* pInput, ma_uint32 frameCount)
{
    if (!pInput) return;

    const size_t sampleCount = frameCount * m_channels;

    // Call user callback if set
    if (m_audioCallback) {
        m_audioCallback(pInput, frameCount);
    }

    // Store in ring buffer
    {
        std::lock_guard<std::mutex> lock(m_bufferMutex);

        for (size_t i = 0; i < sampleCount; ++i) {
            m_audioBuffer[m_writePos] = pInput[i];
            m_writePos = (m_writePos + 1) % m_bufferSize;
        }

        m_hasNewData = true;
    }
}
