#include "Renderer.h"
#include <imgui.h>
#include <glog/logging.h>

void Renderer::drawGui()
{
    static float reloadMessageTimer = 0.0f;
    static bool reloadSuccess = false;
    static float kernelReloadMessageTimer = 0.0f;
    static bool kernelReloadSuccess = false;

    ImGui::Separator();
    ImGui::Text("Postprocessing");

    float exposure = getExposure();
    if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 3.0f)) {
        setExposure(exposure);
    }

    float bloomStrength = getBloomStrength();
    if (ImGui::SliderFloat("Bloom", &bloomStrength, 0.0f, 0.2f)) {
        setBloomStrength(bloomStrength);
    }

    ImGui::Separator();
    ImGui::Text("Shader");

    if (ImGui::Button("Reload Shaders (R)"))
    {
        reloadSuccess = reloadShaders();
        reloadMessageTimer = 3.0f; // Show message for 3 seconds

        if (reloadSuccess)
        {
            LOG(INFO) << "✓ Shaders reloaded successfully from GUI";
        }
        else
        {
            LOG(ERROR) << "✗ Failed to reload shaders - check logs for details";
        }
    }

    ImGui::SameLine();

    // Display GPU performance metrics
    ImGui::Separator();
    ImGui::Text("GPU Performance");

    float execTime = getRaymarcherExecutionTimeMs();
    ImGui::Text("Raymarch Time: %.3f ms", execTime);

    GLint workGroupX = 0, workGroupY = 0, workGroupZ = 0;
    getRaymarcherWorkGroupSize(workGroupX, workGroupY, workGroupZ);
    ImGui::Text("Work Group Size: %dx%dx%d", workGroupX, workGroupY, workGroupZ);

    // Calculate approximate number of threads
    if (workGroupX > 0 && workGroupY > 0 && workGroupZ > 0)
    {
        int totalThreadsPerGroup = workGroupX * workGroupY * workGroupZ;
        ImGui::Text("Threads/Group: %d", totalThreadsPerGroup);
    }
}
