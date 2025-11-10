#pragma once

#include "app/Screen.h"
#include <imgui.h>
#include "Camera.h"
#include "Renderer.h"
#include <memory>

class MainScreen : public IScreen
{
public:
    void onAttach(App &app) override;
    void onResize(int width, int height) override;
    void onUpdate(double dt) override;
    void onRender() override;
    void onDetach() override;
    void onMouseButton(int button, int action, int mods, vec2 px) override;
    void onCursorPos(vec2 px) override;
    void onScroll(double xoffset, double yoffset, vec2 px) override;
    void onGui() override;

private:
    App *m_app{nullptr};
    Camera m_camera{};
    Renderer m_renderer{};
    InteractionController m_interaction{};
    SceneModel m_scene{};

    float m_currentFPS{0.0f};
};
