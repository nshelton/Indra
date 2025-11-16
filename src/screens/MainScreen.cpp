#include "MainScreen.h"
#include "app/App.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <glog/logging.h>
#include <core/core.h>

void MainScreen::onAttach(App &app)
{
    google::InitGoogleLogging("Indra");
    google::SetStderrLogging(google::GLOG_INFO);

    m_app = &app;
    m_scene.reset();

    // Initialize audio capture and analyzer
    if (m_audioCapture.initialize(44100, 2)) {
        LOG(INFO) << "Audio capture initialized successfully";
        if (m_audioAnalyzer.initialize(2048)) {
            LOG(INFO) << "Audio analyzer initialized successfully";
            m_audioAnalyzer.getFrequencyBins(m_frequencyBins, 44100);
        }
    }

    // Set up automatic shader hot-reload
    m_shaderWatcher.watch("../../shaders/pointcloud.vert.glsl", [this](const std::string& path) {
        LOG(INFO) << "Detected change in " << path << ", reloading shaders...";
        m_renderer.reloadShaders();
    });
    m_shaderWatcher.watch("../../shaders/pointcloud.frag.glsl", [this](const std::string& path) {
        LOG(INFO) << "Detected change in " << path << ", reloading shaders...";
        m_renderer.reloadShaders();
    });
    LOG(INFO) << "Shader auto-reload enabled ";
}

void MainScreen::onResize(int width, int height)
{
    m_renderer.setSize(width, height);
    m_camera.setSize(width, height);
    m_interaction.setScreenSize(vec2(static_cast<float>(width), static_cast<float>(height)));
}

void MainScreen::onUpdate(double dt)
{
    m_currentFPS = m_currentFPS * 0.9f + 0.1f * static_cast<float>(1.0 / dt);

    // Check for shader file changes and auto-reload
    m_shaderWatcher.update();

    // Handle keyboard-driven actions that should work outside of ImGui widgets
    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_Delete))
    {
       // handle keypresses
    }

    // Reload shaders with R key (manual override)
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_R))
    {
        m_renderer.reloadShaders();
    }

    // Process audio if enabled
    if (m_audioEnabled && m_audioCapture.isCapturing()) {
        if (m_audioCapture.getLatestAudioData(m_audioBuffer)) {
            // Perform FFT analysis on the audio data
            // This computes FFT on GPU and downloads magnitudes to CPU for GUI display
            m_audioAnalyzer.analyzeStereo(m_audioBuffer, m_fftMagnitudes);

            // // Use GPU buffer directly for audio-reactive visuals (avoids CPU round-trip!)
            // // The magnitudes are already on GPU from the analyze call
            // if (!m_fftMagnitudes.empty()) {
            //     m_renderer.setFFTDataGPU(m_audioAnalyzer.getDeviceMagnitudes(),
            //                              m_audioAnalyzer.getNumBins());

            //     // Example: Find dominant frequency (uses CPU copy for GUI)
            //     auto maxIt = std::max_element(m_fftMagnitudes.begin(), m_fftMagnitudes.end());
            //     size_t maxIdx = std::distance(m_fftMagnitudes.begin(), maxIt);
            //     float dominantFreq = m_frequencyBins[maxIdx];
            //     // LOG(INFO) << "Dominant frequency: " << dominantFreq << " Hz";
            // }
        }
    }
}

void MainScreen::onRender()
{
    m_renderer.render(m_camera, m_scene, m_interaction.state());
}

void MainScreen::onDetach()
{
    m_renderer.shutdown();
    // Clean up any resources here
}

void MainScreen::onMouseButton(int button, int action, int /*mods*/, vec2 px)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        m_interaction.onMouseDown(m_scene, m_camera, px);
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        m_interaction.onMouseUp();
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
    {
        // Middle click always pans the page; never selects/moves entities
        m_interaction.beginPan(m_camera, px);
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE)
    {
        m_interaction.onMouseUp();
    }
}

void MainScreen::onCursorPos(vec2 px)
{
    m_interaction.onCursorPos(m_scene, m_camera, px);
}

void MainScreen::onScroll(double xoffset, double yoffset, vec2 px)
{
    m_interaction.onScroll(m_scene, m_camera, static_cast<float>(yoffset), px);
}
