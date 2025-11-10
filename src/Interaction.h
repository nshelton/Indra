#pragma once

#include <optional>
#include "core/core.h"
#include "Camera.h"
#include "Scene.h"
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

    void updateHover(const SceneModel &scene, const Camera &camera, const vec2 &mousePx);
    void onMouseDown(SceneModel &scene, Camera &camera, const vec2 &px);
    // Begin a camera pan irrespective of what's under the cursor
    void beginPan(Camera &camera, const vec2 &px);
    void onMouseUp();
    void onCursorPos(SceneModel &scene, Camera &camera, const vec2 &px);
    void onScroll(SceneModel &scene, Camera &camera, float yoffset, const vec2 &px);

    const InteractionState &state() const { return m_state; }

    std::optional<int> HoveredEntity() const { return m_state.hoveredId; }
    std::optional<int> SelectedEntity() const { return m_state.activeId; }

    void SelectEntity(int id) { m_state.activeId = id; }
    void ClearHover() { m_state.hoveredId.reset(); }
    void DeselectEntity() { m_state.activeId.reset(); }

    bool ShowPathNodes() const { return m_state.showPathNodes; }
    void SetShowPathNodes(bool v) { m_state.showPathNodes = v; }

private:
    InteractionState m_state;
    TrackballController m_trackball;
    vec2 m_lastMousePos{0, 0};
    vec2 m_screenSize{1920, 1080};
};