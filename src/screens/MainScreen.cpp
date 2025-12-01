#include "MainScreen.h"
#include "app/App.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <glog/logging.h>
#include <core/core.h>
#include <util/Serialization.h>

#include "TrackballController.h"
#include "FirstPersonController.h"

static std::string STATE_FILE = "project_state.json";

void MainScreen::onAttach(App &app)
{
    google::InitGoogleLogging("Indra");
    google::SetStderrLogging(google::GLOG_INFO);

    m_app = &app;
    m_camera = Camera(vec3(10.0f, 10.0f, 10.0f), vec3(0.0f, 0.0f, 0.0f));

    // Initialize renderer now that OpenGL context is ready
    m_renderer.init();

    createCameraController(m_cameraControlType);

    // Watch the include directory for changes to included shader files
    m_shaderWatcher.watchDirectory("../../shaders/", [this](const std::string &path)
                                   {
        LOG(INFO) << "Detected change in shader " << path << ", reloading shaders...";
        m_renderer.reloadShaders(); }, true); // recursive = true to watch subdirectories

    LOG(INFO) << "Shader auto-reload enabled ";

    // serialize at the end
    std::string error_string;
    bool loaded = serialization::loadState(m_camera, m_renderer, STATE_FILE, &error_string);
}

void MainScreen::createCameraController(CameraControlType type) {
    if (type == CameraControlType::Trackball) {
        m_cameraController = std::make_unique<TrackballController>();
    } else {
        m_cameraController = std::make_unique<FirstPersonController>();
    }
    m_cameraController->setCamera(&m_camera);
    m_cameraControlType = type;
}


void MainScreen::onResize(int width, int height)
{
    m_renderer.setSize(width, height);
    m_camera.setSize(width, height);
}

void MainScreen::onUpdate(double dt)
{
    m_currentFPS = m_currentFPS * 0.9f + 0.1f * static_cast<float>(1.0 / dt);

    // Check for shader file changes and auto-reload
    m_shaderWatcher.update();

    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_R))
    {
        m_renderer.reloadShaders();
    }

    if (m_cameraController) {
        m_cameraController->update(static_cast<float>(dt));
    }
}

void MainScreen::onRender()
{
    m_renderer.render(m_camera);
}

void MainScreen::onDetach()
{
    // Save current state before shutting down
    std::string error_string;
    bool saved = serialization::saveState(m_camera, m_renderer, STATE_FILE, &error_string);

    // Clean up any resources here
    m_renderer.shutdown();
}

void MainScreen::(int button, int action, int mods, vec2 px)
{onMouseButton
    if (m_cameraController) {
        m_cameraController->onMouseButton(button, action, mods);
    }
}

void MainScreen::onCursorPos(vec2 px)
{
    if (m_cameraController) {
        m_cameraController->onCursorPos(px.x, px.y);
    }
}

void MainScreen::onScroll(double xoffset, double yoffset, vec2 px)
{
    if (m_cameraController) {
        m_cameraController->onMouseWheel(xoffset, yoffset);
    }
}