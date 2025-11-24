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

    m_raymarcher->drawGui();
}
