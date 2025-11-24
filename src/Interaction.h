#pragma once

#include <optional>
#include "core/core.h"
#include "Camera.h"
#include "TrackballController.h"

#define HANDLE_HITBOX_RADIUS 10.0f

enum class InteractionMode
{
    None,
    Pan,
    Rotate,
    ResizingEntity
};


struct InteractionState
{
    std::optional<int> hoveredId;
    std::optional<int> activeId; // the entity we grabbed
    InteractionMode mode = InteractionMode::None;

    // Rendering options
    bool showPathNodes{true};
};

class InteractionController
{
public:
    void setScreenSize(const vec2& size) { m_screenSize = size; }

    void updateHover(const Camera &camera, const vec2 &mousePx);
    void onMouseDown(Camera &camera, const vec2 &px);
    // Begin a camera pan irrespective of what's under the cursor
    void beginPan(Camera &camera, const vec2 &px);
    void onMouseUp();
    void onCursorPos(Camera &camera, const vec2 &px);
    void onScroll(Camera &camera, float yoffset, const vec2 &px);

    // Keyboard movement
    void moveKeyboard(Camera &camera, float deltaTime,
                      bool forward, bool back, bool left, bool right, bool down, bool up)
    {
        m_trackball.moveKeyboard(camera, deltaTime, forward, back, left, right, down, up);
    }

    const InteractionState &state() const { return m_state; }

    void drawGUI();

private:
    InteractionState m_state;
    TrackballController m_trackball;
    vec2 m_lastMousePos{0, 0};
    vec2 m_screenSize{1920, 1080};
};