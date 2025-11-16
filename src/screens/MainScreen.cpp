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
    m_scene.initScene();
    m_renderer.setPoints(m_scene.points(), color::white());

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
    m_shaderWatcher.watch("../../kernels/pointcloud_kernel.cu", [this](const std::string& path) {
        LOG(INFO) << "Detected change in " << path << ", reloading CUDA kernel...";
        m_renderer.reloadKernel();
    });
    // Also watch the source kernel file and sync it to kernels/ directory
    m_shaderWatcher.watch("../../src/render/PointCloudKernel.cu", [this](const std::string& path) {
        LOG(INFO) << "Detected change in source kernel " << path << ", transforming for NVRTC and reloading...";

        // Read source file
        std::ifstream src(path);
        if (!src) {
            LOG(ERROR) << "Failed to open source kernel file: " << path;
            return;
        }

        // Read line by line and transform for NVRTC
        std::stringstream nvrtcContent;
        std::string line;
        bool skipHostFunction = false;
        bool addedParticleData = false;

        while (std::getline(src, line)) {
            // Skip include lines
            if (line.find("#include \"PointCloudKernel.cuh\"") != std::string::npos) {
                // Add ParticleData struct definition instead of include
                if (!addedParticleData) {
                    nvrtcContent << "// Particle data structure for physics simulation\n";
                    nvrtcContent << "struct ParticleData\n";
                    nvrtcContent << "{\n";
                    nvrtcContent << "    float vx, vy, vz;     // Velocity (vec3)\n";
                    nvrtcContent << "    float age;            // Current age in seconds\n";
                    nvrtcContent << "    float maxAge;         // Maximum age before respawn\n";
                    nvrtcContent << "    float pressure;       // Pressure value\n";
                    nvrtcContent << "    float u, v;           // UV coordinates\n";
                    nvrtcContent << "};\n\n";
                    addedParticleData = true;
                }
                continue;
            }

            // Skip other includes that NVRTC can't handle
            if (line.find("#include <cuda_runtime.h>") != std::string::npos ||
                line.find("#include <device_launch_parameters.h>") != std::string::npos ||
                line.find("#include <cmath>") != std::string::npos) {
                continue;
            }

            // Add extern "C" to kernel function
            if (line.find("__global__ void animatePointsKernel") != std::string::npos) {
                nvrtcContent << "extern \"C\" " << line << "\n";
                continue;
            }

            // Skip host function (starts with "cudaError_t launchAnimatePointsKernel")
            if (line.find("cudaError_t launchAnimatePointsKernel") != std::string::npos) {
                skipHostFunction = true;
            }

            if (!skipHostFunction) {
                nvrtcContent << line << "\n";
            }
        }
        src.close();

        std::string transformedContent = nvrtcContent.str();

        // Write to kernels directory (try both source and build locations)
        std::ofstream dst1("../../kernels/pointcloud_kernel.cu");
        std::ofstream dst2("kernels/pointcloud_kernel.cu");

        if (dst1) {
            dst1 << transformedContent;
            dst1.close();
        }
        if (dst2) {
            dst2 << transformedContent;
            dst2.close();
        }

        // Reload the kernel
        m_renderer.reloadKernel();
    });
    LOG(INFO) << "Shader and kernel auto-reload enabled - edit files and they'll reload automatically!";
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

            // Use GPU buffer directly for audio-reactive visuals (avoids CPU round-trip!)
            // The magnitudes are already on GPU from the analyze call
            if (!m_fftMagnitudes.empty()) {
                m_renderer.setFFTDataGPU(m_audioAnalyzer.getDeviceMagnitudes(),
                                         m_audioAnalyzer.getNumBins());

                // Example: Find dominant frequency (uses CPU copy for GUI)
                auto maxIt = std::max_element(m_fftMagnitudes.begin(), m_fftMagnitudes.end());
                size_t maxIdx = std::distance(m_fftMagnitudes.begin(), maxIt);
                float dominantFreq = m_frequencyBins[maxIdx];
                // LOG(INFO) << "Dominant frequency: " << dominantFreq << " Hz";
            }
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
    LOG(INFO) << "MouseDown at pixel (" << px.x << ", " << px.y << ")";

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
