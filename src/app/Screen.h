#pragma once

class App;
#include "core/core.h"

class IScreen {
public:
    virtual ~IScreen() = default;
    virtual void onAttach(App& app) = 0;
    virtual void onResize(int width, int height) = 0;
    virtual void onUpdate(double dt) = 0;
    virtual void onRender() = 0;
    virtual void onDetach() {}

    // Input (optional overrides)
    virtual void onMouseButton(int /*button*/, int /*action*/, int /*mods*/, vec2 /*pixel_pos*/) {}
    virtual void onCursorPos(vec2 /*pixel_pos*/) {}
    virtual void onScroll(double /*xoffset*/, double /*yoffset*/, vec2 /*pixel_pos*/) {}

    // UI (ImGui) hook
    virtual void onGui() {}
};
