#pragma once

#include "core/Core.h"
#include "render/LineRenderer.h"
#include "render/MeshRenderer.h"
#include "render/PointCloudRenderer.h"
#include "Interaction.h"
#include "Scene.h"
#include "Camera.h"

#include <glog/logging.h>

class Renderer
{
public:
    Renderer();

    void setSize(int width, int height)
    {
        LOG(INFO) << "GL size set to " << width << "x" << height;
        glViewport(0, 0, width, height);
    }

    void render(const Camera &camera, const SceneModel &scene, const InteractionState &uiState);
    void setPoints(const std::vector<vec3> &points, color col);
    void shutdown();

    int totalVertices() const { return static_cast<int>(m_lines.totalVertices()); }

private:
    LineRenderer m_lines{};
    MeshRenderer m_meshes{};
    PointCloudRenderer m_points{};
    float m_time{0.0f};
};
