#include "Camera.h"
#include <imgui.h>

void Camera::drawGui() const
{
    ImGui::Separator();
    ImGui::BeginGroup();
    ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", m_position.x, m_position.y, m_position.z);
    ImGui::Text("Camera Target: (%.2f, %.2f, %.2f)", m_target.x, m_target.y, m_target.z);
    ImGui::EndGroup();

    ImGui::Text("Projection Matrix:");
    matrix4 proj = getProjectionMatrix();
    for (int i = 0; i < 4; ++i)
    {
        ImGui::Text("| %.3f %.3f %.3f %.3f |", proj.m[i][0], proj.m[i][1], proj.m[i][2], proj.m[i][3]);
    }

    ImGui::Text("Camera Up: (%.2f, %.2f, %.2f)", m_up.x, m_up.y, m_up.z);
    ImGui::Text("Camera Right: (%.2f, %.2f, %.2f)", m_right.x, m_right.y, m_right.z);
    ImGui::Text("Camera Forward: (%.2f, %.2f, %.2f)", m_forward.x, m_forward.y, m_forward.z);
}
