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
    ImGui::Text("HDR Postprocessing");

    float exposure = getExposure();
    if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 3.0f)) {
        setExposure(exposure);
    }

    float bloomStrength = getBloomStrength();
    if (ImGui::SliderFloat("Bloom Strength", &bloomStrength, 0.0f, 0.2f)) {
        setBloomStrength(bloomStrength);
    }

    float grainAmount = getGrainAmount();
    if (ImGui::SliderFloat("Film Grain", &grainAmount, 0.0f, 0.1f)) {
        setGrainAmount(grainAmount);
    }

    ImGui::Separator();
    ImGui::Text("Shaders & Kernels");

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

    if (ImGui::Button("Reload Kernel (K)"))
    {
        kernelReloadSuccess = reloadKernel();
        kernelReloadMessageTimer = 3.0f; // Show message for 3 seconds

        if (kernelReloadSuccess)
        {
            LOG(INFO) << "✓ CUDA kernel reloaded successfully from GUI";
        }
        else
        {
            LOG(ERROR) << "✗ Failed to reload CUDA kernel - check logs for details";
        }
    }

    // Show shader reload notification
    if (reloadMessageTimer > 0.0f)
    {
        reloadMessageTimer -= ImGui::GetIO().DeltaTime;

        if (reloadSuccess)
        {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Shaders reloaded successfully!");
        }
        else
        {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Shader reload failed! Check console.");
        }
    }

    // Show kernel reload notification
    if (kernelReloadMessageTimer > 0.0f)
    {
        kernelReloadMessageTimer -= ImGui::GetIO().DeltaTime;

        if (kernelReloadSuccess)
        {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "CUDA kernel reloaded successfully!");
        }
        else
        {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Kernel reload failed! Check console.");
        }
    }
}
