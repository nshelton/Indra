#include "Scene.h"

void SceneModel::initScene()
{
    // make some random boxes to start
    for (int i = 0; i < 100; ++i)
    {
        boxMesh box(vec3(5, 5, 5));
        box.transform = transform(
            vec3(static_cast<float>(rand() % 200 - 100),
                 static_cast<float>(rand() % 200 - 100),
                 static_cast<float>(rand() % 200 - 100)),
            quaternion::fromAxisAngle(vec3(0, 1, 0), static_cast<float>(rand() % 6280 / 1000.0f)),
            vec3(1, 1, 1));
        addMesh(box);
    }

    m_points = std::vector<vec3>();

    for (int i = 0; i < m_numPoints; ++i)
    {
        m_points.push_back(vec3(
            static_cast<float>(rand() % 200 - 100),
            static_cast<float>(rand() % 200 - 100),
            static_cast<float>(rand() % 200 - 100)));
    }
}