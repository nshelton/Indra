#pragma once

#include "core/vec2.h"

class Camera;

class ICameraController {
public:
    virtual ~ICameraController() = default;

    virtual void onCursorPos(double xpos, double ypos) = 0;
    virtual void onMouseButton(int button, int action, int mods) = 0;
    virtual void onMouseWheel(double xoffset, double yoffset) = 0;
    virtual void update(float dt) = 0;
    virtual void drawGui() = 0;
    virtual void setCamera(Camera* camera) = 0;
};
