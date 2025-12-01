#pragma once

#include "core/core.h"
#include "ICameraController.h"

class Camera;

// Trackball camera controller similar to Three.js TrackballControls
// Allows rotating, panning, and zooming a camera around a target point
class TrackballController : public ICameraController
{
public:
    TrackballController();

    void onCursorPos(double xpos, double ypos) override;
    void onMouseButton(int button, int action, int mods) override;
    void onMouseWheel(double xoffset, double yoffset) override;
    void update(float dt) override;
    void drawGui() override;
    void setCamera(Camera* camera) override { m_camera = camera; }

    // Settings
    void setRotateSpeed(float speed) { m_rotateSpeed = speed; }
    void setPanSpeed(float speed) { m_panSpeed = speed; }
    void setZoomSpeed(float speed) { m_zoomSpeed = speed; }
    void setKeyboardSpeed(float speed) { m_keyboardSpeed = speed; }
    void setMinDistance(float dist) { m_minDistance = dist; }
    void setMaxDistance(float dist) { m_maxDistance = dist; }

    float* rotateSpeed() { return &m_rotateSpeed; }
    float* panSpeed() { return &m_panSpeed; }
    float* zoomSpeed() { return &m_zoomSpeed; }
    float* keyboardSpeed() { return &m_keyboardSpeed; }

private:
    // Project a 2D screen point onto a 3D sphere for trackball rotation
    vec3 projectToSphere(const vec2& point, const vec2& screenSize) const;
    void rotate(const vec2& delta, const vec2& screenSize);
    void pan(const vec2& delta, const vec2& screenSize);
    void zoom(float delta);
    void moveKeyboard(float dt);

    Camera* m_camera{nullptr};
    vec2 m_lastMousePos{};
    bool m_rotating{false};
    bool m_panning{false};

    float m_rotateSpeed{1.0f};
    float m_panSpeed{1.0f};
    float m_zoomSpeed{1.2f};
    float m_keyboardSpeed{0.5f}; // Base keyboard movement speed (scales with distance)
    float m_minDistance{0.1f};
    float m_maxDistance{1000.0f};
    float m_trackballRadius{0.8f}; // Radius of virtual trackball (0-1 range)
};
