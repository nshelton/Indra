#include "AudioCapture.h"
#include "miniaudio.h"
#include <glog/logging.h>
#include <algorithm>

// PIMPL implementation - contains miniaudio types
struct AudioCapture::Impl {
    ma_device device;
    ma_device_config deviceConfig;

    // Static callback for miniaudio
    static void dataCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);
};

void AudioCapture::Impl::dataCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    AudioCapture* pCapture = static_cast<AudioCapture*>(pDevice->pUserData);
    if (pCapture && pInput)
    {
        pCapture->onAudioData(static_cast<const float*>(pInput), frameCount);
    }

    // For loopback mode, we don't output anything
    (void)pOutput;
}

AudioCapture::AudioCapture()
    : m_impl(std::make_unique<Impl>())
    , m_sampleRate(44100)
    , m_channels(2)
    , m_isCapturing(false)
    , m_hasInitialized(false)
{
    // Pre-allocate buffer for 2048 samples * 2 channels
    m_audioBuffer.resize(2048 * 2, 0.0f);
}

AudioCapture::~AudioCapture()
{
    stop();
    if (m_hasInitialized)
    {
        ma_device_uninit(&m_impl->device);
    }
}

bool AudioCapture::initialize(unsigned int sampleRate, unsigned int channels)
{
    m_sampleRate = sampleRate;
    m_channels = channels;

    // Resize buffer to match 2048 samples * channels for FFT
    m_audioBuffer.resize(2048 * channels, 0.0f);

    // Configure device for loopback capture
    m_impl->deviceConfig = ma_device_config_init(ma_device_type_loopback);
    m_impl->deviceConfig.capture.format = ma_format_f32; // 32-bit float
    m_impl->deviceConfig.capture.channels = channels;
    m_impl->deviceConfig.sampleRate = sampleRate;
    m_impl->deviceConfig.dataCallback = &AudioCapture::Impl::dataCallback;
    m_impl->deviceConfig.pUserData = this;

    // Initialize the device
    ma_result result = ma_device_init(NULL, &m_impl->deviceConfig, &m_impl->device);
    if (result != MA_SUCCESS)
    {
        LOG(ERROR) << "Failed to initialize audio capture device: " << result;
        return false;
    }

    LOG(INFO) << "Audio capture initialized: " << sampleRate << "Hz, " << channels << " channels";
    return true;
}

bool AudioCapture::start()
{
    if (m_isCapturing)
    {
        LOG(WARNING) << "Audio capture already running";
        return false;
    }

    ma_result result = ma_device_start(&m_impl->device);
    if (result != MA_SUCCESS)
    {
        LOG(ERROR) << "Failed to start audio capture: " << result;
        return false;
    }

    m_isCapturing = true;
    LOG(INFO) << "Audio capture started";
    m_hasInitialized = true;
    return true;
}

void AudioCapture::stop()
{
    if (!m_isCapturing)
    {
        return;
    }

    ma_device_stop(&m_impl->device);
    m_isCapturing = false;
    LOG(INFO) << "Audio capture stopped";
}

bool AudioCapture::getLatestAudioData(std::vector<float> &outBuffer)
{
    // Always return current buffer for real-time visualization
    // Don't wait for m_hasNewData flag
    std::lock_guard<std::mutex> lock(m_bufferMutex);
    outBuffer = m_audioBuffer;
    return true;
}

void AudioCapture::setAudioCallback(std::function<void(const float *data, unsigned int frameCount)> callback)
{
    m_audioCallback = callback;
}

void AudioCapture::onAudioData(const float *pInput, unsigned int frameCount)
{
    if (!pInput)
        return;

    // Call user callback if set
    if (m_audioCallback)
    {
        m_audioCallback(pInput, frameCount);
    }

    // Copy latest audio chunk directly to buffer
    const size_t sampleCount = frameCount * m_channels;
    const size_t copyCount = std::min(sampleCount, m_audioBuffer.size());

    {
        std::lock_guard<std::mutex> lock(m_bufferMutex);
        std::copy(pInput, pInput + copyCount, m_audioBuffer.begin());
    }
}
