#pragma once
#include <vector>
#include "core/core.h"
#include "Scene.h"

class SceneModel
{
public:
    void initScene();
    void clear()
    {
        // Clear the scene data
    }

    void addMesh(const mesh &m)
    {
        m_meshes.push_back(m);
    }

    const std::vector<mesh> &meshes() const { return m_meshes; }
    const std::vector<vec3> &points() const { return m_points; }

    bool needsUpload() const { return m_needsUpload; }
    void setNeedsUpload(bool value) { m_needsUpload = value; }

    int m_numPoints = 1024 * 1024;


private:
    std::vector<mesh> m_meshes;
    std::vector<vec3> m_points;
    bool m_needsUpload = false;
};