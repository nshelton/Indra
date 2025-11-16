#include "Renderer.h"

#include <algorithm>
#include <cmath>

Renderer::Renderer()
{
   // OpenGL resources will be initialized in init()
   m_hdrFramebuffer = std::make_unique<Framebuffer>();
   m_postProcessor = std::make_unique<PostProcessor>();
}

void Renderer::init()
{
   LOG(INFO) << "Initializing renderer OpenGL resources";
   m_lines.init();
   m_raymarcher.init();
   TextureBlit::init();
   m_initialized = true;
}

void Renderer::render(const Camera &camera, const ShaderState &shaderState, const InteractionState &uiState)
{
   if (!m_initialized) {
      LOG(WARNING) << "Renderer not initialized, call init() first";
      return;
   }

   if (!m_hdrFramebuffer || m_hdrFramebuffer->getFBO() == 0) {
      LOG(WARNING) << "HDR framebuffer not ready, call setSize() first";
      return;
   }

   m_lines.clear();

   // Bind HDR framebuffer and render scene to it
   m_hdrFramebuffer->bind();

   // Clear the HDR buffer
   glClearColor(0.f, 0.f, 0.f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   // Run CUDA kernel to animate/process points on GPU (before drawing)
   m_time += 0.016f;  // Increment animation time (~60 FPS)

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

   // Raymarch to texture (doesn't draw to framebuffer directly)
   m_raymarcher.draw(camera, shaderState);

   // Blit raymarcher output to HDR framebuffer
   TextureBlit::blit(m_raymarcher.getOutputTexture());

   // Unbind HDR framebuffer
   m_hdrFramebuffer->unbind();

   // Apply HDR postprocessing (bloom, noise grain, tone mapping) to screen
   m_postProcessor->process(m_hdrFramebuffer->getColorTexture());
}

void Renderer::shutdown()
{
   m_lines.shutdown();
   m_raymarcher.shutdown();
   TextureBlit::shutdown();

   // HDR resources will be cleaned up automatically by unique_ptr destructors
   m_hdrFramebuffer.reset();
   m_postProcessor.reset();
}