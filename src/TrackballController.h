#pragma once

#include "core/core.h"
#include "Camera.h"

// Trackball camera controller similar to Three.js TrackballControls
// Allows rotating, panning, and zooming a camera around a target point
class TrackballController
{
public:
    TrackballController();

    // Update camera based on mouse drag for rotation
    void rotate(Camera& camera, const vec2& delta, const vec2& screenSize);

    // Pan camera and target in screen space
    void pan(Camera& camera, const vec2& delta, const vec2& screenSize);

    // Zoom camera (move closer/further from target)
    void zoom(Camera& camera, float delta);

    // Keyboard movement (flythrough style)
    // deltaTime is in seconds, direction flags indicate which way to move
    void moveKeyboard(Camera& camera, float deltaTime,
                      bool forward, bool back, bool left, bool right, bool down, bool up);

    // Settings
    void setRotateSpeed(float speed) { m_rotateSpeed = speed; }
    void setPanSpeed(float speed) { m_panSpeed = speed; }
    void setZoomSpeed(float speed) { m_zoomSpeed = speed; }
    void setKeyboardSpeed(float speed) { m_keyboardSpeed = speed; }
    void setMinDistance(float dist) { m_minDistance = dist; }
    void setMaxDistance(float dist) { m_maxDistance = dist; }

    float rotateSpeed() const { return m_rotateSpeed; }
    float panSpeed() const { return m_panSpeed; }
    float zoomSpeed() const { return m_zoomSpeed; }
    float keyboardSpeed() const { return m_keyboardSpeed; }

private:
    // Project a 2D screen point onto a 3D sphere for trackball rotation
    vec3 projectToSphere(const vec2& point, const vec2& screenSize) const;

    float m_rotateSpeed{1.0f};
    float m_panSpeed{1.0f};
    float m_zoomSpeed{1.2f};
    float m_keyboardSpeed{0.5f}; // Base keyboard movement speed (scales with distance)
    float m_minDistance{0.1f};
    float m_maxDistance{1000.0f};
    float m_trackballRadius{0.8f}; // Radius of virtual trackball (0-1 range)
};
