#include "MainScreen.h"
#include <string>
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

    ImGui::End();
}
