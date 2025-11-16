#include "Camera.h"
#include <imgui.h>

void Camera::drawGui()
{
    ImGui::Separator();
    ImGui::BeginGroup();
    ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", m_position.x, m_position.y, m_position.z);
    ImGui::Text("Camera Target: (%.2f, %.2f, %.2f)", m_target.x, m_target.y, m_target.z);
    ImGui::EndGroup();

    if (ImGui::Button("Reset Camera"))
    {
        m_position = vec3(0, 0, 5);
        m_target = vec3(0, 0, 0);
        m_up = vec3(0, 1, 0);
        m_viewDirty = true;
        updateViewMatrix();
    }
}
