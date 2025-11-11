#include "AudioAnalyzer.h"
#include <glog/logging.h>
#include <cmath>
#include <algorithm>

AudioAnalyzer::AudioAnalyzer()
    : m_fftSize(2048)
    , m_plan(0)
    , d_input(nullptr)
    , d_output(nullptr)
    , m_initialized(false)
{
}

AudioAnalyzer::~AudioAnalyzer()
{
    if (m_initialized) {
        cufftDestroy(m_plan);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
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

    // Create cuFFT plan for real-to-complex 1D FFT
    cufftResult result = cufftPlan1d(&m_plan, m_fftSize, CUFFT_R2C, 1);
    if (result != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to create cuFFT plan: " << result;
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    m_initialized = true;
    LOG(INFO) << "AudioAnalyzer initialized with FFT size: " << m_fftSize;
    return true;
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

    // Copy audio data to device
    cudaError_t cudaStatus = cudaMemcpy(d_input, audioData.data(),
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

    // Wait for FFT to complete
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "CUDA synchronization failed: " << cudaGetErrorString(cudaStatus);
        return false;
    }

    // Copy results back to host and compute magnitudes
    const unsigned int numBins = m_fftSize / 2 + 1;
    std::vector<cufftComplex> complexOutput(numBins);

    cudaStatus = cudaMemcpy(complexOutput.data(), d_output,
                            numBins * sizeof(cufftComplex),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to copy FFT results to host: " << cudaGetErrorString(cudaStatus);
        return false;
    }

    // Compute magnitudes on CPU
    magnitudes.resize(numBins);
    for (unsigned int i = 0; i < numBins; ++i) {
        float real = complexOutput[i].x;
        float imag = complexOutput[i].y;
        magnitudes[i] = std::sqrt(real * real + imag * imag);
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
