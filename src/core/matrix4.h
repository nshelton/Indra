#pragma once

#include "vec3.h"
#include "quaternion.h"

struct matrix4
{
    float m[4][4];

    matrix4()
    {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    static matrix4 translation(const vec3 &t)
    {
        matrix4 result;
        result.m[0][3] = t.x;
        result.m[1][3] = t.y;
        result.m[2][3] = t.z;
        return result;
    }
    static matrix4 scale(const vec3 &s)
    {
        matrix4 result;
        result.m[0][0] = s.x;
        result.m[1][1] = s.y;
        result.m[2][2] = s.z;
        return result;
    }
    static matrix4 rotation(const quaternion &q)
    {
        matrix4 result;
        float xx = q.x * q.x;
        float yy = q.y * q.y;
        float zz = q.z * q.z;
        float xy = q.x * q.y;
        float xz = q.x * q.z;
        float yz = q.y * q.z;
        float wx = q.w * q.x;
        float wy = q.w * q.y;
        float wz = q.w * q.z;

        result.m[0][0] = 1.0f - 2.0f * (yy + zz);
        result.m[0][1] = 2.0f * (xy - wz);
        result.m[0][2] = 2.0f * (xz + wy);
        result.m[1][0] = 2.0f * (xy + wz);
        result.m[1][1] = 1.0f - 2.0f * (xx + zz);
        result.m[1][2] = 2.0f * (yz - wx);
        result.m[2][0] = 2.0f * (xz - wy);
        result.m[2][1] = 2.0f * (yz + wx);
        result.m[2][2] = 1.0f - 2.0f * (xx + yy);
        return result;
    }

    static matrix4 TRS(const vec3 &translation, const quaternion &rotation, const vec3 &scale)
    {
        return matrix4::translation(translation) * matrix4::rotation(rotation) * matrix4::scale(scale);
    }

    vec3 multiplyPoint(const vec3 &point) const
    {
        float x = m[0][0] * point.x + m[0][1] * point.y + m[0][2] * point.z + m[0][3];
        float y = m[1][0] * point.x + m[1][1] * point.y + m[1][2] * point.z + m[1][3];
        float z = m[2][0] * point.x + m[2][1] * point.y + m[2][2] * point.z + m[2][3];
        float w = m[3][0] * point.x + m[3][1] * point.y + m[3][2] * point.z + m[3][3];
        if (w != 0.0f)
        {
            x /= w;
            y /= w;
            z /= w;
        }
        return vec3(x, y, z);
    }

    matrix4 operator*(const matrix4 &other) const
    {
        matrix4 result;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                result.m[i][j] = 0.0f;
                for (int k = 0; k < 4; ++k)
                {
                    result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }

    matrix4 inverse() const
    {
        // transpose upper 3x3
        matrix4 result;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                result.m[i][j] = m[j][i];
            }
        }
        // inverse translation
        result.m[0][3] = -(result.m[0][0] * m[0][3] + result.m[0][1] * m[1][3] + result.m[0][2] * m[2][3]);
        result.m[1][3] = -(result.m[1][0] * m[0][3] + result.m[1][1] * m[1][3] + result.m[1][2] * m[2][3]);
        result.m[2][3] = -(result.m[2][0] * m[0][3] + result.m[2][1] * m[1][3] + result.m[2][2] * m[2][3]);
        result.m[3][3] = 1.0f;
        return result;
    }
};