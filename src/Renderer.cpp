#include "Renderer.h"

#include <algorithm>
#include <cmath>

Renderer::Renderer()
{
   m_lines.init();
   m_meshes.init();
}

void Renderer::render(const Camera &camera, const SceneModel &scene, const InteractionState &uiState)
{
   m_lines.clear();

   // Clear the screen
   glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   // Render all meshes in the scene
   for (const auto& m : const_cast<SceneModel&>(scene).meshes())
   {
      m_meshes.renderMesh(m, camera);
   }

   // Draw ground plane grid
   vec3 gridCenter(0, 0, 0);
   float gridSize = 200.0f;
   int gridDivisions = 10;
   color gridColor(0.3f, 0.3f, 0.3f, 1.0f);    // Dark gray for grid lines
   color centerColor(0.5f, 0.5f, 0.5f, 1.0f);  // Lighter gray for center lines
   m_lines.addGrid(gridCenter, gridSize, gridDivisions, gridColor, centerColor);

   // Draw 3D axis lines for reference (above the grid)
   float axisLength = 50.0f;
   m_lines.addLine(vec3(0, 0.1f, 0), vec3(axisLength, 0.1f, 0), color(1, 0, 0, 1)); // X axis - Red
   m_lines.addLine(vec3(0, 0.1f, 0), vec3(0, axisLength, 0.1f), color(0, 1, 0, 1)); // Y axis - Green
   m_lines.addLine(vec3(0, 0.1f, 0), vec3(0, 0.1f, axisLength), color(0, 0, 1, 1)); // Z axis - Blue

   // Render lines (for debug/UI elements)
   m_lines.draw(camera);
}

void Renderer::shutdown()
{
   m_lines.shutdown();
   m_meshes.shutdown();
}