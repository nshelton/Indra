#include "TrackballController.h"
#include <cmath>
#include <algorithm>
#include <imgui.h>
#include "core/core.h"
#include "core/vec2.h"
#include "Camera.h"


TrackballController::TrackballController() {}

void TrackballController::onCursorPos(double xpos, double ypos) {
    if (!m_camera) return;

    const vec2 mousePos(static_cast<float>(xpos), static_cast<float>(ypos));
    const vec2 delta = mousePos - m_lastMousePos;
    // TODO: get screen size from somewhere. App::getScreenSize() or something. For now, hardcode.
    const vec2 screenSize(1280, 720);

    if (m_rotating) {
        rotate(delta, screenSize);
    }
    if (m_panning) {
        pan(delta, screenSize);
    }

    m_lastMousePos = mousePos;
}

void TrackballController::onMouseButton(int button, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    if (button == 0 && action == 1) { // Left mouse down
        m_rotating = true;
    } else if (button == 0 && action == 0) { // Left mouse up
        m_rotating = false;
    } else if (button == 1 && action == 1) { // Right mouse down
        m_panning = true;
    } else if (button == 1 && action == 0) { // Right mouse up
        m_panning = false;
    }
}

void TrackballController::onMouseWheel(double xoffset, double yoffset) {
    if (!m_camera) return;
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    zoom(static_cast<float>(yoffset));
}

void TrackballController::update(float dt) {
    if (!m_camera) return;
    
    moveKeyboard(dt);
}

void TrackballController::moveKeyboard(float dt) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard) return;

    bool forward = ImGui::IsKeyDown(ImGuiKey_W);
    bool back = ImGui::IsKeyDown(ImGuiKey_S);
    bool left = ImGui::IsKeyDown(ImGuiKey_A);
    bool right = ImGui::IsKeyDown(ImGuiKey_D);
    bool down = ImGui::IsKeyDown(ImGuiKey_Q);
    bool up = ImGui::IsKeyDown(ImGuiKey_E);

    // Calculate movement direction
    vec3 moveDirection(0.0f, 0.0f, 0.0f);

    if (forward) moveDirection = moveDirection + m_camera->getForward();
    if (back) moveDirection = moveDirection - m_camera->getForward();
    if (left) moveDirection = moveDirection - m_camera->getRight();
    if (right) moveDirection = moveDirection + m_camera->getRight();
    if (down) moveDirection = moveDirection - vec3(0,1,0); // Use world up for down
    if (up) moveDirection = moveDirection + vec3(0,1,0); // Use world up for up

    // Early exit if no movement
    if (moveDirection.length() < 0.001f)
        return;

    // Calculate distance from camera to target for zoom-based scaling
    vec3 position = m_camera->getPosition();
    vec3 target = m_camera->getTarget();
    vec3 offset = position - target;
    float distance = offset.length();

    // Scale movement speed based on distance (similar to pan and zoom)
    float scaledSpeed = m_keyboardSpeed * distance * dt;

    // Apply movement
    moveDirection = moveDirection.normalized() * scaledSpeed;
    m_camera->setPosition(position + moveDirection);
    m_camera->setTarget(target + moveDirection);
}


void TrackballController::drawGui() {
   ImGui::Text("Trackball Settings");
   ImGui::SliderFloat("Rotate Speed", &m_rotateSpeed, 0.01f, 10.0f);
   ImGui::SliderFloat("Pan Speed", &m_panSpeed, 0.01f, 10.0f);
   ImGui::SliderFloat("Zoom Speed", &m_zoomSpeed, 0.1f, 5.0f);
   ImGui::SliderFloat("Keyboard Speed", &m_keyboardSpeed, 0.1f, 5.0f);
}

vec3 TrackballController::projectToSphere(const vec2& point, const vec2& screenSize) const
{
    // Normalize screen coordinates to [-1, 1] range
    float x = (2.0f * point.x / screenSize.x) - 1.0f;
    float y = 1.0f - (2.0f * point.y / screenSize.y); // Flip Y axis

    float length = x * x + y * y;
    float z = 0.0f;

    if (length <= m_trackballRadius * m_trackballRadius * 0.5f)
    {
        // Inside sphere
        z = std::sqrt(m_trackballRadius * m_trackballRadius - length);
    }
    else
    {
        // Outside sphere - use hyperbolic sheet
        z = (m_trackballRadius * m_trackballRadius * 0.5f) / std::sqrt(length);
    }

    return vec3(x, y, z).normalized();
}

void TrackballController::rotate(const vec2& delta, const vec2& screenSize)
{
    if (!m_camera) return;
    if (delta.x == 0.0f && delta.y == 0.0f)
        return;

    // Get camera properties
    vec3 position = m_camera->getPosition();
    vec3 target = m_camera->getTarget();

    // Calculate offset from target to camera
    vec3 offset = position - target;

    // Horizontal rotation angle (around world Y axis)
    float angleX = -delta.x * m_rotateSpeed * 0.005f;

    // Vertical rotation angle
    float angleY = -delta.y * m_rotateSpeed * 0.005f;

    // First, apply horizontal rotation around world Y axis
    quaternion rotY = quaternion::fromAxisAngle(vec3(0, 1, 0), angleX);
    quaternion offsetQuat = quaternion(0, offset.x, offset.y, offset.z);
    quaternion rotYConj = quaternion(rotY.w, -rotY.x, -rotY.y, -rotY.z);
    quaternion rotatedY = rotY * offsetQuat * rotYConj;
    vec3 offsetAfterY(rotatedY.x, rotatedY.y, rotatedY.z);

    // For vertical rotation, compute right vector in XZ plane
    // This ensures the up vector stays as world Y
    vec3 forward = (target - (target + offsetAfterY)).normalized();
    vec3 worldUp(0, 1, 0);
    vec3 right = forward.cross(worldUp);

    // If right vector is too small (near pole), use a safe fallback
    float rightLength = right.length();
    if (rightLength < 0.01f)
    {
        // Near the pole - use XZ plane directly to compute a stable right vector
        right = vec3(-offsetAfterY.z, 0, offsetAfterY.x);
        rightLength = right.length();
        if (rightLength < 0.01f)
        {
            // Fallback to a fixed right vector if still unstable
            right = vec3(1, 0, 0);
        }
        else
        {
            right = right / rightLength;
        }
    }
    else
    {
        right = right / rightLength;
    }

    // Clamp vertical angle to prevent gimbal lock
    // Calculate current elevation angle
    vec3 horizontalDir(offsetAfterY.x, 0, offsetAfterY.z);
    float horizontalDist = horizontalDir.length();

    float currentElevation = std::atan2(offsetAfterY.y, horizontalDist);
    float newElevation = currentElevation + angleY;

    // Clamp to prevent going over the poles (leaving safe margin to avoid gimbal lock)
    const float maxElevation = 1.4f; // About 80 degrees - prevents gimbal lock and numerical instability
    newElevation = std::max(-maxElevation, std::min(maxElevation, newElevation));
    angleY = newElevation - currentElevation;

    // Skip rotation if we're too close to the pole (prevents glitchy behavior)
    if (std::abs(newElevation) >= maxElevation - 0.01f && std::abs(angleY) < 0.001f)
    {
        angleY = 0.0f;
    }

    // Apply vertical rotation around the right vector
    quaternion rotX = quaternion::fromAxisAngle(right, angleY);
    quaternion offsetQuatY = quaternion(0, offsetAfterY.x, offsetAfterY.y, offsetAfterY.z);
    quaternion rotXConj = quaternion(rotX.w, -rotX.x, -rotX.y, -rotX.z);
    quaternion rotatedFinal = rotX * offsetQuatY * rotXConj;

    // Set new camera position
    vec3 newOffset(rotatedFinal.x, rotatedFinal.y, rotatedFinal.z);
    m_camera->setPosition(target + newOffset);
    m_camera->setTarget(target);
    m_camera->setUp(worldUp); // Always set up to world Y
}

void TrackballController::pan(const vec2& delta, const vec2& screenSize)
{
    if (!m_camera) return;
    if (delta.x == 0.0f && delta.y == 0.0f)
        return;

    vec3 position = m_camera->getPosition();
    vec3 target = m_camera->getTarget();
    vec3 offset = position - target;
    float distance = offset.length();

    // Pan speed scales with distance from target
    float panScale = distance * m_panSpeed * 0.001f;

    // Use camera's right vector for horizontal panning
    vec3 right = m_camera->getRight();

    // Use world Y for vertical panning to keep movement intuitive
    vec3 worldUp(0, 1, 0);

    vec3 panOffset = right * (-delta.x * panScale) + worldUp * (delta.y * panScale);

    m_camera->setPosition(position + panOffset);
    m_camera->setTarget(target + panOffset);
    m_camera->setUp(worldUp); // Always set up to world Y
}

void TrackballController::zoom(float delta)
{
    if (!m_camera) return;
    if (delta == 0.0f)
        return;

    vec3 position = m_camera->getPosition();
    vec3 target = m_camera->getTarget();

    vec3 offset = position - target;
    float distance = offset.length();

    // Zoom by moving along view direction
    float zoomAmount = delta * m_zoomSpeed * distance * 0.05f;
    float newDistance = distance - zoomAmount;

    // Clamp distance
    newDistance = std::max(m_minDistance, std::min(m_maxDistance, newDistance));

    // Set new position
    vec3 direction = offset.normalized();
    m_camera->setPosition(target + direction * newDistance);
    m_camera->setTarget(target);
}