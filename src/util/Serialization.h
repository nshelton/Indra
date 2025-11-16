#pragma once

#include <string>

struct ShaderState;
class Camera;
class Renderer;

namespace serialization {

// Save/Load full project (page + camera + renderer + plotter config) to JSON file.
bool saveState(const ShaderState& model, const Camera& camera, const Renderer& renderer,
                 const std::string& filePath, std::string* errorOut = nullptr);
bool loadState(ShaderState& model, Camera& camera, Renderer& renderer,
                 const std::string& filePath, std::string* errorOut = nullptr);

}


