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

    std::vector<mesh> &meshes() { return m_meshes; }

private:
    std::vector<mesh> m_meshes;
};