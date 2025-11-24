#include "Serialization.h"

#include <fstream>
#include <sstream>
#include <filesystem>

#include <nlohmann/json.hpp>

#include "core/Core.h"

#include "Camera.h"
#include "Renderer.h"


using nlohmann::json;

namespace serialization
{

    static constexpr int kSchemaVersion = 2;

    bool saveState(const Camera &camera, const Renderer &renderer,
                   const std::string &filePath, std::string *errorOut)
    {
        json j;
        j["schemaVersion"] = kSchemaVersion;

        // state.toJson(j["shaderState"]);
        camera.toJson(j["camera"]);
        renderer.toJson(j["renderer"]);

        std::ofstream file(filePath);
        if (!file.is_open())
        {
            if (errorOut) *errorOut = "Failed to open file for writing: " + filePath;
            return false;
        }
        file << j.dump(4);
        return true;
    }

    bool loadState(Camera &camera, Renderer &renderer,
                   const std::string &filePath, std::string *errorOut)
    {
        std::ifstream file(filePath);
        if (!file.is_open())
        {
            if (errorOut) *errorOut = "Failed to open file for reading: " + filePath;
            return false;
        }

        json j;
        try
        {
            file >> j;
        }
        catch (const std::exception &e)
        {
            if (errorOut) *errorOut = "Failed to parse JSON: " + std::string(e.what());
            return false;
        }

        int schemaVersion = j.value("schemaVersion", 0);
        if (schemaVersion != kSchemaVersion)
        {
            if (errorOut) *errorOut = "Unsupported schema version: " + std::to_string(schemaVersion);
            return false;
        }

        // state.fromJson(j["shaderState"]);
        camera.fromJson(j["camera"]);
        renderer.fromJson(j["renderer"]);

        return true;
    }

} // namespace serialization
