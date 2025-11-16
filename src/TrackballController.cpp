#include "TrackballController.h"
#include <cmath>
#include <algorithm>

TrackballController::TrackballController()
{
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

void TrackballController::rotate(Camera& camera, const vec2& delta, const vec2& screenSize)
{
    if (delta.x == 0.0f && delta.y == 0.0f)
        return;

    // Get camera properties
    vec3 position = camera.getPosition();
    vec3 target = camera.getTarget();

    // Calculate offset from target to camera
    vec3 offset = position - target;
    float radius = offset.length();

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
    camera.setPosition(target + newOffset);
    camera.setTarget(target);
    camera.setUp(worldUp); // Always set up to world Y
}

void TrackballController::pan(Camera& camera, const vec2& delta, const vec2& screenSize)
{
    if (delta.x == 0.0f && delta.y == 0.0f)
        return;

    vec3 position = camera.getPosition();
    vec3 target = camera.getTarget();
    vec3 offset = position - target;
    float distance = offset.length();

    // Pan speed scales with distance from target
    float panScale = distance * m_panSpeed * 0.001f;

    // Use camera's right vector for horizontal panning
    vec3 right = camera.getRight();

    // Use world Y for vertical panning to keep movement intuitive
    vec3 worldUp(0, 1, 0);

    vec3 panOffset = right * (-delta.x * panScale) + worldUp * (delta.y * panScale);

    camera.setPosition(position + panOffset);
    camera.setTarget(target + panOffset);
    camera.setUp(worldUp); // Always set up to world Y
}

void TrackballController::zoom(Camera& camera, float delta)
{
    if (delta == 0.0f)
        return;

    vec3 position = camera.getPosition();
    vec3 target = camera.getTarget();

    vec3 offset = position - target;
    float distance = offset.length();

    // Zoom by moving along view direction
    float zoomAmount = delta * m_zoomSpeed * distance * 0.05f;
    float newDistance = distance - zoomAmount;

    // Clamp distance
    newDistance = std::max(m_minDistance, std::min(m_maxDistance, newDistance));

    // Set new position
    vec3 direction = offset.normalized();
    camera.setPosition(target + direction * newDistance);
    camera.setTarget(target);
}

void TrackballController::moveKeyboard(Camera& camera, float deltaTime,
                                       bool forward, bool back, bool left, bool right, bool down, bool up)
{
    // Calculate movement direction
    vec3 moveDirection(0.0f, 0.0f, 0.0f);

    if (forward) moveDirection = moveDirection + camera.getForward();
    if (back) moveDirection = moveDirection - camera.getForward();
    if (left) moveDirection = moveDirection - camera.getRight();
    if (right) moveDirection = moveDirection + camera.getRight();
    if (down) moveDirection = moveDirection - camera.getUp();
    if (up) moveDirection = moveDirection + camera.getUp();

    // Early exit if no movement
    if (moveDirection.length() < 0.001f)
        return;

    // Calculate distance from camera to target for zoom-based scaling
    vec3 position = camera.getPosition();
    vec3 target = camera.getTarget();
    vec3 offset = position - target;
    float distance = offset.length();

    // Scale movement speed based on distance (similar to pan and zoom)
    // Using a smaller multiplier for more controlled movement
    float scaledSpeed = m_keyboardSpeed * distance * deltaTime;

    // Apply movement
    moveDirection = moveDirection.normalized() * scaledSpeed;
    camera.setPosition(position + moveDirection);
    camera.setTarget(target + moveDirection);
}
