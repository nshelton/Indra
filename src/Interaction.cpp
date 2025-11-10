#include "Interaction.h"

#include "Scene.h"
#include "Camera.h"
#include "core/core.h"
#include <glog/logging.h>
#include <cmath>

void InteractionController::updateHover(const SceneModel &scene, const Camera &camera, const vec2 &mouseWorld)
{
    // TODO: Implement hover detection for scene entities
}

void InteractionController::onMouseDown(SceneModel &scene, Camera &camera, const vec2 &px)
{
    m_lastMousePos = px;
    m_state.mode = InteractionMode::Rotate;
}

void InteractionController::beginPan(Camera &camera, const vec2 &px)
{
    m_lastMousePos = px;
    m_state.mode = InteractionMode::Pan;
}

void InteractionController::onCursorPos(SceneModel &scene, Camera &camera, const vec2 &px)
{
    vec2 delta = px - m_lastMousePos;

    if (m_state.mode == InteractionMode::Rotate)
    {
        m_trackball.rotate(camera, delta, m_screenSize);
    }
    else if (m_state.mode == InteractionMode::Pan)
    {
        m_trackball.pan(camera, delta, m_screenSize);
    }

    m_lastMousePos = px;
}

void InteractionController::onScroll(SceneModel &scene, Camera &camera, float yoffset, const vec2 &px)
{
    m_trackball.zoom(camera, yoffset);
}

void InteractionController::onMouseUp()
{
    m_state.mode = InteractionMode::None;
}

