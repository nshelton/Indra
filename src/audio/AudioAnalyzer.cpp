#include "AudioAnalyzer.h"
#include <glog/logging.h>
#include <cmath>
#include <algorithm>

// Forward declare CUDA kernel launcher (defined in AudioAnalyzer.cu)
extern void launchComputeMagnitudes(const cufftComplex* d_complexData, float* d_magnitudes, unsigned int numBins);

AudioAnalyzer::AudioAnalyzer()
    : m_fftSize(2048)
    , m_plan(0)
    , d_input(nullptr)
    , d_output(nullptr)
    , d_magnitudes(nullptr)
    , m_initialized(false)
    , m_windowType(WindowType::None)
{
}

AudioAnalyzer::~AudioAnalyzer()
{
    if (m_initialized) {
        cufftDestroy(m_plan);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_magnitudes) cudaFree(d_magnitudes);
    }
}

bool AudioAnalyzer::initialize(unsigned int fftSize)
{
    // Verify FFT size is power of 2
    if ((fftSize & (fftSize - 1)) != 0) {
        LOG(ERROR) << "FFT size must be a power of 2, got: " << fftSize;
        return false;
    }

    m_fftSize = fftSize;

    // Allocate device memory
    cudaError_t cudaStatus = cudaMalloc(&d_input, m_fftSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate device input buffer: " << cudaGetErrorString(cudaStatus);
        return false;
    }

    // Output is complex, size is (N/2 + 1) for real-to-complex FFT
    cudaStatus = cudaMalloc(&d_output, (m_fftSize / 2 + 1) * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate device output buffer: " << cudaGetErrorString(cudaStatus);
        cudaFree(d_input);
        return false;
    }

    // Allocate device memory for magnitudes
    cudaStatus = cudaMalloc(&d_magnitudes, (m_fftSize / 2 + 1) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate device magnitudes buffer: " << cudaGetErrorString(cudaStatus);
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    // Create cuFFT plan for real-to-complex 1D FFT
    cufftResult result = cufftPlan1d(&m_plan, m_fftSize, CUFFT_R2C, 1);
    if (result != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to create cuFFT plan: " << result;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_magnitudes);
        return false;
    }

    m_initialized = true;

    // Generate window coefficients
    generateWindow();

    LOG(INFO) << "AudioAnalyzer initialized with FFT size: " << m_fftSize;
    return true;
}

bool AudioAnalyzer::reinitialize(unsigned int newFftSize)
{
    // Clean up existing resources
    if (m_initialized) {
        cufftDestroy(m_plan);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_magnitudes) cudaFree(d_magnitudes);
        d_input = nullptr;
        d_output = nullptr;
        d_magnitudes = nullptr;
        m_initialized = false;
    }

    // Initialize with new FFT size
    return initialize(newFftSize);
}

void AudioAnalyzer::generateWindow()
{
    m_window.resize(m_fftSize);
    const double pi = 3.14159265358979323846;

    switch (m_windowType) {
        case WindowType::None:
            // Rectangular window (all ones)
            std::fill(m_window.begin(), m_window.end(), 1.0f);
            break;

        case WindowType::Hann:
            // Hann window: 0.5 * (1 - cos(2*pi*n/(N-1)))
            for (unsigned int i = 0; i < m_fftSize; ++i) {
                m_window[i] = 0.5f * (1.0f - std::cos(2.0 * pi * i / (m_fftSize - 1)));
            }
            break;

        case WindowType::BlackmanHarris:
            // 4-term Blackman-Harris window for excellent sidelobe suppression (-92 dB)
            // Reduces spectral leakage artifacts significantly
            for (unsigned int i = 0; i < m_fftSize; ++i) {
                const double a0 = 0.35875;
                const double a1 = 0.48829;
                const double a2 = 0.14128;
                const double a3 = 0.01168;
                const double phase = 2.0 * pi * i / (m_fftSize - 1);

                m_window[i] = a0
                            - a1 * std::cos(phase)
                            + a2 * std::cos(2.0 * phase)
                            - a3 * std::cos(3.0 * phase);
            }
            break;
    }
}

void AudioAnalyzer::setWindowType(WindowType type)
{
    m_windowType = type;
    if (m_initialized) {
        generateWindow();
    }
}

bool AudioAnalyzer::analyze(const std::vector<float>& audioData, std::vector<float>& magnitudes)
{
    if (!m_initialized) {
        LOG(ERROR) << "AudioAnalyzer not initialized";
        return false;
    }

    if (audioData.size() < m_fftSize) {
        LOG(WARNING) << "Audio data size (" << audioData.size() << ") less than FFT size (" << m_fftSize << ")";
        return false;
    }

    // Apply window function to audio data
    std::vector<float> windowedData(m_fftSize);
    for (unsigned int i = 0; i < m_fftSize; ++i) {
        windowedData[i] = audioData[i] * m_window[i];
    }

    // Copy windowed audio data to device
    cudaError_t cudaStatus = cudaMemcpy(d_input, windowedData.data(),
                                         m_fftSize * sizeof(float),
                                         cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to copy audio data to device: " << cudaGetErrorString(cudaStatus);
        return false;
    }

    // Execute FFT
    cufftResult result = cufftExecR2C(m_plan, d_input, d_output);
    if (result != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to execute FFT: " << result;
        return false;
    }

    // Compute magnitudes on GPU
    const unsigned int numBins = m_fftSize / 2 + 1;
    launchComputeMagnitudes(d_output, d_magnitudes, numBins);

    // Wait for magnitude computation to complete
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "CUDA synchronization failed: " << cudaGetErrorString(cudaStatus);
        return false;
    }

    // Copy magnitudes to host for GUI display (if needed)
    // The magnitudes remain on GPU (d_magnitudes) for direct use by other kernels
    magnitudes.resize(numBins);
    cudaStatus = cudaMemcpy(magnitudes.data(), d_magnitudes,
                            numBins * sizeof(float),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to copy magnitudes to host: " << cudaGetErrorString(cudaStatus);
        return false;
    }

    return true;
}

bool AudioAnalyzer::analyzeStereo(const std::vector<float>& stereoData, std::vector<float>& magnitudes)
{
    if (stereoData.size() < m_fftSize * 2) {
        LOG(WARNING) << "Stereo data size insufficient for FFT";
        return false;
    }

    // Convert stereo to mono by averaging channels
    std::vector<float> monoData(m_fftSize);
    for (unsigned int i = 0; i < m_fftSize; ++i) {
        monoData[i] = (stereoData[i * 2] + stereoData[i * 2 + 1]) * 0.5f;
    }

    return analyze(monoData, magnitudes);
}

void AudioAnalyzer::getFrequencyBins(std::vector<float>& frequencies, unsigned int sampleRate)
{
    const unsigned int numBins = m_fftSize / 2 + 1;
    frequencies.resize(numBins);

    const float binWidth = static_cast<float>(sampleRate) / static_cast<float>(m_fftSize);
    for (unsigned int i = 0; i < numBins; ++i) {
        frequencies[i] = i * binWidth;
    }
}
