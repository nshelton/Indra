#pragma once

#include "app/Screen.h"
#include <imgui.h>
#include "Camera.h"
#include "Renderer.h"
#include "ShaderState.h"
#include "audio/AudioCapture.h"
#include "audio/AudioAnalyzer.h"
#include "util/FileWatcher.h"
#include <memory>

class MainScreen : public IScreen
{
public:
    void onAttach(App &app) override;
    void onResize(int width, int height) override;
    void onUpdate(double dt) override;
    void onRender() override;
    void onDetach() override;
    void onMouseButton(int button, int action, int mods, vec2 px) override;
    void onCursorPos(vec2 px) override;
    void onScroll(double xoffset, double yoffset, vec2 px) override;
    void onGui() override;

private:
    void drawAudioGui();

    App *m_app{nullptr};
    Camera m_camera{};
    Renderer m_renderer{};
    InteractionController m_interaction{};
    ShaderState m_shaderState{};

    float m_currentFPS{0.0f};

    // Audio analysis
    AudioCapture m_audioCapture{};
    AudioAnalyzer m_audioAnalyzer{};
    std::vector<float> m_audioBuffer{};
    std::vector<float> m_fftMagnitudes{};
    std::vector<float> m_frequencyBins{};
    bool m_audioEnabled{false};
    int m_windowTypeIndex{0}; // Default to BlackmanHarris
    int m_fftSizeIndex{2}; // Default to 2048 (index 2 in [512, 1024, 2048, 4096, 8192])

    // Shader hot-reload
    FileWatcher m_shaderWatcher{};
};
