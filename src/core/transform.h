#pragma once

#include "vec3.h"
#include "quaternion.h"
#include "matrix4.h"

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