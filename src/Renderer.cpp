#include "Renderer.h"

#include <algorithm>
#include <cmath>

Renderer::Renderer()
{
   m_lines.init();
   m_meshes.init();
   m_points.init();
   // CUDA interop will be initialized lazily on first runCudaKernel() call

   // HDR framebuffer and postprocessor will be initialized on first setSize() call
   m_hdrFramebuffer = std::make_unique<Framebuffer>();
   m_postProcessor = std::make_unique<PostProcessor>();
}

void Renderer::setPoints(const std::vector<vec3> &points, color col)
{
   m_points.clear();
   for (const auto &p : points)
   {
      m_points.addPoint(p, col);
   }
   // Upload initial data to GPU
   m_points.uploadToGPU();
}

void Renderer::render(const Camera &camera, const SceneModel &scene, const InteractionState &uiState)
{
   // Check if HDR pipeline exists
   if (!m_hdrFramebuffer || !m_postProcessor) {
      LOG(WARNING) << "HDR pipeline not initialized, skipping render";
      return;
   }

   // Initialize HDR pipeline on first render if size is set
   if (m_width > 0 && m_height > 0 && m_hdrFramebuffer->getFBO() == 0) {
      LOG(INFO) << "Creating HDR framebuffer: " << m_width << "x" << m_height;
      m_hdrFramebuffer->createHDR(m_width, m_height);
      m_postProcessor->init(m_width, m_height);
      LOG(INFO) << "HDR pipeline initialized successfully";
   }

   // Don't render if framebuffer isn't initialized yet
   if (m_hdrFramebuffer->getFBO() == 0) {
      LOG(WARNING) << "HDR framebuffer FBO is 0, skipping render (width=" << m_width << ", height=" << m_height << ")";
      return;
   }

   m_lines.clear();

   // Bind HDR framebuffer and render scene to it
   m_hdrFramebuffer->bind();

   // Clear the HDR buffer
   glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   // Render all meshes in the scene
   for (const auto& m : const_cast<SceneModel&>(scene).meshes())
   {
      m_meshes.renderMesh(m, camera);
   }

   // Run CUDA kernel to animate/process points on GPU (before drawing)
   m_time += 0.016f;  // Increment animation time (~60 FPS)
   m_points.runCudaKernel(m_time);

   m_points.draw(camera);

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

   // Unbind HDR framebuffer
   m_hdrFramebuffer->unbind();

   // Apply HDR postprocessing (bloom, noise grain, tone mapping) to screen
   m_postProcessor->process(m_hdrFramebuffer->getColorTexture());
}

void Renderer::shutdown()
{
   m_lines.shutdown();
   m_meshes.shutdown();
   m_points.shutdown();

   // HDR resources will be cleaned up automatically by unique_ptr destructors
   m_hdrFramebuffer.reset();
   m_postProcessor.reset();
}