#pragma once
#include <vector>
#include "core/core.h"
#include "Scene.h"

class Scene
{
public:
    void reset()
    {
        backgroundColor = vec3{0.0f, 0.0f, 0.0f};
        spherePosition = vec3{0.0f, 0.0f, 0.0f};
        sphereRadius = 1.0f;
    }

private:
    vec3 backgroundColor{0.0f, 0.0f, 0.0f};
    vec3 spherePosition{0.0f, 0.0f, 0.0f};
    float sphereRadius{1.0f};
};