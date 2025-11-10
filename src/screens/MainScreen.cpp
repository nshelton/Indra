#include "MainScreen.h"
#include "app/App.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <glog/logging.h>
#include <core/core.h>

void MainScreen::onAttach(App &app)
{
    google::InitGoogleLogging("Indra");
    google::SetStderrLogging(google::GLOG_INFO);

    m_app = &app;
    m_scene.initScene();
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
    // Handle keyboard-driven actions that should work outside of ImGui widgets
    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_Delete))
    {
       // handle keypresses
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
