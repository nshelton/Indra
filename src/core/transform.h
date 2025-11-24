#pragma once

#include "vec3.h"
#include "quaternion.h"
#include "matrix4.h"
#include <nlohmann/json_fwd.hpp>

struct transform
{
    vec3 position;
    quaternion rotation;
    vec3 scale;

    transform() : position(0,0,0), rotation(), scale(1,1,1) {}
    transform(const vec3 &pos, const quaternion &rot, const vec3 &scl)
        : position(pos), rotation(rot), scale(scl) {}

    vec3 apply(const vec3 &point) const;
    matrix4 toMatrix() const;
};

inline vec3 transform::apply(const vec3 &point) const {
    // Scale
    vec3 p = point * scale;
    // Rotate
    quaternion pQuat(0, p.x, p.y, p.z);
    quaternion rConj = quaternion(rotation.w, -rotation.x, -rotation.y, -rotation.z);
    quaternion rotated = rotation * pQuat * rConj;
    // Translate
    return vec3(
        rotated.x + position.x,
        rotated.y + position.y,
        rotated.z + position.z
    );
}

inline matrix4 transform::toMatrix() const {
    return matrix4::TRS(position, rotation, scale);
}

// JSON serialization
inline void to_json(nlohmann::json& j, const transform& t)
{
    j = nlohmann::json{{"position", t.position}, {"rotation", t.rotation}, {"scale", t.scale}};
}

inline void from_json(const nlohmann::json& j, transform& t)
{
    if (j.contains("position")) j.at("position").get_to(t.position);
    if (j.contains("rotation")) j.at("rotation").get_to(t.rotation);
    if (j.contains("scale")) j.at("scale").get_to(t.scale);
}