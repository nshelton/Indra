#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>

// Color stores floats but in the range [0,1] for RGBA components
struct color
{
    float r, g, b, a;

    color() : r(1), g(1), b(1), a(1) {}
    color(float r, float g, float b, float a = 1.0f) : r(r), g(g), b(b), a(a) {}

    static color fromHex(uint32_t hexValue)
    {
        float r = ((hexValue >> 16) & 0xFF) / 255.0f;
        float g = ((hexValue >> 8) & 0xFF) / 255.0f;
        float b = (hexValue & 0xFF) / 255.0f;
        return color(r, g, b, 1.0f);
    }

    static color white() { return color(1, 1, 1, 1); }
    static color black() { return color(0, 0, 0, 1); }
    static color red() { return color(1, 0, 0, 1); }
    static color green() { return color(0, 1, 0, 1); }
    static color blue() { return color(0, 0, 1, 1); }
};

// JSON serialization
inline void to_json(nlohmann::json& j, const color& c)
{
    j = nlohmann::json{{"r", c.r}, {"g", c.g}, {"b", c.b}, {"a", c.a}};
}

inline void from_json(const nlohmann::json& j, color& c)
{
    c.r = j.value("r", 1.0f);
    c.g = j.value("g", 1.0f);
    c.b = j.value("b", 1.0f);
    c.a = j.value("a", 1.0f);
}