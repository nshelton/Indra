#include "FirstPersonController.h"
#include "Camera.h"
#include "core/core.h"
#include <imgui.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
    inline float radians(float degrees) {
        return static_cast<float>(degrees * M_PI / 180.0f);
    }

    inline float degrees(float radians) {
        return static_cast<float>(radians * 180.0f / M_PI);
    }
}

FirstPersonController::FirstPersonController() {}

void FirstPersonController::setCamera(Camera* camera) {
    m_camera = camera;
    if (m_camera) {
        vec3 dir = (m_camera->getTarget() - m_camera->getPosition()).normalized();
        m_yaw = degrees(atan2(dir.z, dir.x));
        m_pitch = degrees(asin(dir.y));
    }
}

void FirstPersonController::onCursorPos(double xpos, double ypos) {
    if (!m_camera || !m_look) return;

    const vec2 mousePos(static_cast<float>(xpos), static_cast<float>(ypos));
    if (m_lastMousePos.x == 0 && m_lastMousePos.y == 0) {
        m_lastMousePos = mousePos; // First event after look enabled
        return;
    }
    const vec2 delta = mousePos - m_lastMousePos;
    m_lastMousePos = mousePos;

    m_yaw += delta.x * m_lookSpeed;
    m_pitch -= delta.y * m_lookSpeed;

    // Clamp pitch
    m_pitch = clamp(m_pitch, -89.0f, 89.0f);

    vec3 front;
    front.x = cos(radians(m_yaw)) * cos(radians(m_pitch));
    front.y = sin(radians(m_pitch));
    front.z = sin(radians(m_yaw)) * cos(radians(m_pitch));
    m_camera->setTarget(m_camera->getPosition() + front.normalized());
}

void FirstPersonController::onMouseButton(int button, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    // Right mouse button to look
    if (button == 0 && action == 1) {
        if (!m_look) {
            m_look = true;
            m_lastMousePos = vec2(0,0); // a jump
            // TODO: Hide and center cursor from MainScreen/App
        }
    } else if (button == 0 && action == 0) {
        if (m_look) {
            m_look = false;
            // TODO: Show cursor from MainScreen/App
        }
    }
}

void FirstPersonController::onMouseWheel(double xoffset, double yoffset) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    m_moveSpeed += static_cast<float>(yoffset) * 0.5f;
    m_moveSpeed = clamp(m_moveSpeed, 0.1f, 100.0f);
}

void FirstPersonController::update(float dt) {
    if (!m_camera) return;

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard) return;

    vec3 move(0.0f);
    if (ImGui::IsKeyDown(ImGuiKey_W)) move = move + m_camera->getForward();
    if (ImGui::IsKeyDown(ImGuiKey_S)) move = move - m_camera->getForward();
    if (ImGui::IsKeyDown(ImGuiKey_A)) move = move - m_camera->getRight();
    if (ImGui::IsKeyDown(ImGuiKey_D)) move = move + m_camera->getRight();
    if (ImGui::IsKeyDown(ImGuiKey_E)) move = move + vec3(0, 1, 0);
    if (ImGui::IsKeyDown(ImGuiKey_Q)) move = move - vec3(0, 1, 0);

    if (move.length() > 0.001f) {
        vec3 pos = m_camera->getPosition();
        vec3 target = m_camera->getTarget();
        vec3 delta = move.normalized() * m_moveSpeed * dt;
        m_camera->setPosition(pos + delta);
        m_camera->setTarget(target + delta);
    }
}

void FirstPersonController::drawGui() {
    ImGui::Text("First Person Settings");
    ImGui::SliderFloat("Look Speed", &m_lookSpeed, 0.01f, 1.0f);
    ImGui::SliderFloat("Move Speed", &m_moveSpeed, 0.1f, 20.0f);
}
