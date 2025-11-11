#pragma once

#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>

class AudioAnalyzer {
public:
    AudioAnalyzer();
    ~AudioAnalyzer();

    // Initialize FFT with specified size (must be power of 2)
    bool initialize(unsigned int fftSize = 2048);

    // Perform FFT on audio data
    // Input: time-domain audio samples (mono)
    // Output: frequency-domain magnitudes
    bool analyze(const std::vector<float>& audioData, std::vector<float>& magnitudes);

    // Analyze stereo audio (average the channels first)
    bool analyzeStereo(const std::vector<float>& stereoData, std::vector<float>& magnitudes);

    // Get frequency bins
    void getFrequencyBins(std::vector<float>& frequencies, unsigned int sampleRate);

    // Get the FFT size
    unsigned int getFFTSize() const { return m_fftSize; }

    // Get number of frequency bins (FFT size / 2 + 1)
    unsigned int getNumBins() const { return m_fftSize / 2 + 1; }

private:
    unsigned int m_fftSize;
    cufftHandle m_plan;

    // Device memory pointers
    float* d_input;         // Input time-domain data
    cufftComplex* d_output; // Output frequency-domain data

    bool m_initialized;

    // Helper to convert complex FFT output to magnitudes
    void computeMagnitudes(cufftComplex* complexData, float* magnitudes, unsigned int size);
};
