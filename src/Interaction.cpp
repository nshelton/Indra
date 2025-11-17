#include "Interaction.h"

#include "ShaderState.h"
#include "Camera.h"
#include "core/core.h"
#include <glog/logging.h>
#include <imgui.h>
#include <cmath>


void InteractionController::onMouseDown(ShaderState &shaderState, Camera &camera, const vec2 &px)
{
    m_lastMousePos = px;
    m_state.mode = InteractionMode::Rotate;
}

void InteractionController::beginPan(Camera &camera, const vec2 &px)
{
    m_lastMousePos = px;
    m_state.mode = InteractionMode::Pan;
}

void InteractionController::onCursorPos(ShaderState &shaderState, Camera &camera, const vec2 &px)
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

void InteractionController::onScroll(ShaderState &shaderState, Camera &camera, float yoffset, const vec2 &px)
{
    m_trackball.zoom(camera, yoffset);
}

void InteractionController::onMouseUp()
{
    m_state.mode = InteractionMode::None;
}

void InteractionController::drawGUI()
{
   ImGui::Text("Interaction Settings");
   ImGui::SliderFloat("Rotate Speed", m_trackball.rotateSpeed(), 0.01f, 10.0f);
   ImGui::SliderFloat("Pan Speed", m_trackball.panSpeed(), 0.01f, 10.0f);
   ImGui::SliderFloat("Zoom Speed", m_trackball.zoomSpeed(), 0.1f, 0.5f);
   ImGui::SliderFloat("Keyboard Speed", m_trackball.keyboardSpeed(), 0.1f, 0.5f);
}

