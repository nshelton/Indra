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

    // Reinitialize with a new FFT size
    bool reinitialize(unsigned int newFftSize);

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

    // Get device pointer to FFT magnitudes (for GPU-to-GPU transfer)
    // Returns nullptr if not initialized or if analysis hasn't been run yet
    float* getDeviceMagnitudes() const { return d_magnitudes; }

    // Window types for spectral analysis
    enum class WindowType {
        None,           // Rectangular (no windowing)
        Hann,          // Good general purpose
        BlackmanHarris // Best sidelobe suppression, recommended for clean tone analysis
    };

    // Set the window type (default: BlackmanHarris)
    void setWindowType(WindowType type);

private:
    void generateWindow();
    unsigned int m_fftSize;
    cufftHandle m_plan;

    // Device memory pointers
    float* d_input;         // Input time-domain data
    cufftComplex* d_output; // Output frequency-domain data
    float* d_magnitudes;    // Output magnitudes (for GPU-to-GPU transfer)

    bool m_initialized;
    WindowType m_windowType;
    std::vector<float> m_window;  // Pre-computed window coefficients

    // Helper to convert complex FFT output to magnitudes
    void computeMagnitudes(cufftComplex* complexData, float* magnitudes, unsigned int size);
};
