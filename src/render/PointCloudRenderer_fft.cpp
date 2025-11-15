// FFT data upload implementation for PointCloudRenderer
// This file contains the uploadFFTData method

#include "PointCloudRenderer.h"
#include <cuda_runtime.h>
#include <iostream>

void PointCloudRenderer::uploadFFTData(const std::vector<float>& fftMagnitudes)
{
    if (!m_cudaInteropInitialized || fftMagnitudes.empty())
    {
        return;  // Silently skip if CUDA not available or no data
    }

    int newNumBins = static_cast<int>(fftMagnitudes.size());

    // Reallocate device buffer if size changed
    if (newNumBins != m_numFFTBins)
    {
        if (m_d_fftData && m_ownsFFTData)
        {
            cudaFree(m_d_fftData);
            m_d_fftData = nullptr;
        }

        cudaError_t err = cudaMalloc(&m_d_fftData, newNumBins * sizeof(float));
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to allocate CUDA memory for FFT data: "
                      << cudaGetErrorString(err) << std::endl;
            m_d_fftData = nullptr;
            m_numFFTBins = 0;
            m_ownsFFTData = false;
            return;
        }

        m_numFFTBins = newNumBins;
        m_ownsFFTData = true;  // We allocated this buffer
    }

    // Copy FFT data from host to device
    cudaError_t err = cudaMemcpy(m_d_fftData, fftMagnitudes.data(),
                                  m_numFFTBins * sizeof(float),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy FFT data to device: "
                  << cudaGetErrorString(err) << std::endl;
    }
}

void PointCloudRenderer::setFFTDataGPU(const float* d_fftData, int numBins)
{
    if (!m_cudaInteropInitialized || !d_fftData || numBins <= 0)
    {
        return;  // Silently skip if CUDA not available or invalid data
    }

    // Free our own buffer if we allocated one previously via uploadFFTData
    if (m_d_fftData && m_ownsFFTData)
    {
        cudaFree(m_d_fftData);
    }

    // Use the external GPU buffer directly (no allocation or copy needed!)
    m_d_fftData = const_cast<float*>(d_fftData);
    m_numFFTBins = numBins;
    m_ownsFFTData = false;  // We don't own this buffer
}
