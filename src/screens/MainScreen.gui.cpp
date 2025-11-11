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

    ImGui::Separator();
    ImGui::BeginGroup();
    ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", m_camera.getPosition().x, m_camera.getPosition().y, m_camera.getPosition().z);
    ImGui::Text("Camera Target: (%.2f, %.2f, %.2f)", m_camera.getTarget().x, m_camera.getTarget().y, m_camera.getTarget().z);
    ImGui::EndGroup();
    ImGui::Text("Projection Matrix:");
    matrix4 proj = m_camera.getProjectionMatrix();
    for (int i = 0; i < 4; ++i)
    {
        ImGui::Text("| %.3f %.3f %.3f %.3f |", proj.m[i][0], proj.m[i][1], proj.m[i][2], proj.m[i][3]);
    }

    ImGui::Text("Camera Up: (%.2f, %.2f, %.2f)", m_camera.getUp().x, m_camera.getUp().y, m_camera.getUp().z);
    ImGui::Text("Camera Right: (%.2f, %.2f, %.2f)", m_camera.getRight().x, m_camera.getRight().y, m_camera.getRight().z);
    ImGui::Text("Camera Forward: (%.2f, %.2f, %.2f)", m_camera.getForward().x, m_camera.getForward().y, m_camera.getForward().z);

    // Audio Analysis Section
    ImGui::Separator();
    ImGui::Text("Audio Analysis");

    if (ImGui::Checkbox("Enable Audio Capture", &m_audioEnabled)) {
        if (m_audioEnabled) {
            m_audioCapture.start();
        } else {
            m_audioCapture.stop();
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

    ImGui::End();
}
