#include "MainScreen.h"
#include <string>
#include <algorithm>
#include <fmt/format.h>
#include <ctime>

void MainScreen::onGui()
{
    if (ImGui::Begin("Controls"))
    {
        ImGui::Text("Indra Fluid sim");
        ImGui::Text("FPS: %.1f", m_currentFPS);
    }

    // Camera info
    m_camera.drawGui();

    // Renderer controls
    m_renderer.drawGui();

    // Audio controls
    drawAudioGui();

    ImGui::End();
}

void MainScreen::drawAudioGui()
{
    ImGui::Separator();
    ImGui::Text("Audio Analysis");

    if (ImGui::Checkbox("Enable Audio Capture", &m_audioEnabled)) {
        if (m_audioEnabled) {
            m_audioCapture.start();
        } else {
            m_audioCapture.stop();
        }
    }

    // Window function selection
    const char* windowTypes[] = { "None (Rectangular)", "Hann", "Blackman-Harris" };
    if (ImGui::Combo("Window Function", &m_windowTypeIndex, windowTypes, 3)) {
        m_audioAnalyzer.setWindowType(static_cast<AudioAnalyzer::WindowType>(m_windowTypeIndex));
    }

    // FFT size selection
    const char* fftSizes[] = { "512", "1024", "2048", "4096", "8192" };
    const unsigned int fftSizeValues[] = { 512, 1024, 2048, 4096, 8192 };
    if (ImGui::Combo("FFT Size", &m_fftSizeIndex, fftSizes, 5)) {
        unsigned int newSize = fftSizeValues[m_fftSizeIndex];
        if (m_audioAnalyzer.reinitialize(newSize)) {
            // Update frequency bins for the new FFT size
            m_audioAnalyzer.getFrequencyBins(m_frequencyBins, 44100);
        }
    }

    if (m_audioEnabled && !m_fftMagnitudes.empty()) {
        // Find dominant frequency
        auto maxIt = std::max_element(m_fftMagnitudes.begin(), m_fftMagnitudes.end());
        size_t maxIdx = std::distance(m_fftMagnitudes.begin(), maxIt);
        float dominantFreq = m_frequencyBins[maxIdx];

        ImGui::Text("Dominant Frequency: %.1f Hz", dominantFreq);
        ImGui::Text("Magnitude: %.1f", *maxIt);

        // Simple FFT spectrum visualization
        if (ImGui::TreeNode("Frequency Spectrum")) {
            // Display first 512 bins (0-11kHz range approximately)
            const int displayBins =  std::min(512, static_cast<int>(m_fftMagnitudes.size()));

            ImGui::PlotLines("##FFT", m_fftMagnitudes.data(), displayBins,
                           0, "FFT Spectrum", 0.0f, FLT_MAX, ImVec2(0, 100));
            ImGui::TreePop();
        }
    }
}
