#pragma once

#include "ICameraController.h"
#include "core/vec2.h"

class Camera;

class FirstPersonController : public ICameraController {
public:
    FirstPersonController();

    void onCursorPos(double xpos, double ypos) override;
    void onMouseButton(int button, int action, int mods) override;
    void onMouseWheel(double xoffset, double yoffset) override;
    void update(float dt) override;
    void drawGui() override;
    void setCamera(Camera* camera) override;

private:
    void moveKeyboard(float dt);

    Camera* m_camera{nullptr};
    vec2 m_lastMousePos{};
    bool m_look{false};

    float m_lookSpeed{0.1f};
    float m_moveSpeed{5.0f};

    float m_yaw{ -90.0f };
    float m_pitch{ 0.0f };
};
