#pragma once

#include <string>

struct ShaderState;
struct Camera;
class Renderer;

namespace serialization {

// Save/Load full scene state (shaderstate + camera + renderer ) to JSON file.
bool saveState(const ShaderState& model, const Camera& camera, const Renderer& renderer,
                 const std::string& filePath, std::string* errorOut = nullptr);
bool loadState(ShaderState& model, Camera& camera, Renderer& renderer,
                 const std::string& filePath, std::string* errorOut = nullptr);

}


