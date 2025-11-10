#pragma once

#include "transform.h"
#include <vector>
#include "color.h"
#include "vec2.h"
#include "vec3.h"

struct mesh
{
    // Local-to-world transform
    transform transform;

    // Vertex attributes (parallel arrays)
    std::vector<vec3> positions;
    std::vector<vec3> normals; // Added
    std::vector<vec2> uvs;     // Added
    std::vector<color> colors;

    // Indices for indexed rendering
    std::vector<uint32_t> indices;

    // Utility methods
    void clear()
    {
        positions.clear();
        normals.clear();
        uvs.clear();
        colors.clear();
        indices.clear();
    }

    void reserve(size_t vertexCount, size_t indexCount)
    {
        positions.reserve(vertexCount);
        normals.reserve(vertexCount);
        uvs.reserve(vertexCount);
        colors.reserve(vertexCount);
        indices.reserve(indexCount);
    }

    size_t vertexCount() const { return positions.size(); }
    size_t triangleCount() const { return indices.size() / 3; }

    // Helper to add a vertex with all attributes
    void addVertex(const vec3 &pos, const vec3 &normal = vec3(0, 1, 0),
                   const vec2 &uv = vec2(0, 0), const color &col = color::white())
    {
        positions.push_back(pos);
        normals.push_back(normal);
        uvs.push_back(uv);
        colors.push_back(col);
    }

    // Helper to add a triangle
    void addTriangle(uint32_t i0, uint32_t i1, uint32_t i2)
    {
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
    }
};

struct boxMesh : public mesh
{
    boxMesh(const vec3 &dimensions = vec3(1, 1, 1), const color &col = color::white())
    {
        // Pre-allocate for 24 vertices (4 per face * 6 faces) and 36 indices
        reserve(24, 36);

        const float hx = dimensions.x * 0.5f;
        const float hy = dimensions.y * 0.5f;
        const float hz = dimensions.z * 0.5f;

        // Front face (+Z)
        addVertex(vec3(-hx, -hy, hz), vec3(0, 0, 1), vec2(0, 0), col);
        addVertex(vec3(hx, -hy, hz), vec3(0, 0, 1), vec2(1, 0), col);
        addVertex(vec3(hx, hy, hz), vec3(0, 0, 1), vec2(1, 1), col);
        addVertex(vec3(-hx, hy, hz), vec3(0, 0, 1), vec2(0, 1), col);

        // Back face (-Z)
        addVertex(vec3(hx, -hy, -hz), vec3(0, 0, -1), vec2(0, 0), col);
        addVertex(vec3(-hx, -hy, -hz), vec3(0, 0, -1), vec2(1, 0), col);
        addVertex(vec3(-hx, hy, -hz), vec3(0, 0, -1), vec2(1, 1), col);
        addVertex(vec3(hx, hy, -hz), vec3(0, 0, -1), vec2(0, 1), col);

        // Left face (-X)
        addVertex(vec3(-hx, -hy, -hz), vec3(-1, 0, 0), vec2(0, 0), col);
        addVertex(vec3(-hx, -hy, hz), vec3(-1, 0, 0), vec2(1, 0), col);
        addVertex(vec3(-hx, hy, hz), vec3(-1, 0, 0), vec2(1, 1), col);
        addVertex(vec3(-hx, hy, -hz), vec3(-1, 0, 0), vec2(0, 1), col);

        // Right face (+X)
        addVertex(vec3(hx, -hy, hz), vec3(1, 0, 0), vec2(0, 0), col);
        addVertex(vec3(hx, -hy, -hz), vec3(1, 0, 0), vec2(1, 0), col);
        addVertex(vec3(hx, hy, -hz), vec3(1, 0, 0), vec2(1, 1), col);
        addVertex(vec3(hx, hy, hz), vec3(1, 0, 0), vec2(0, 1), col);

        // Bottom face (-Y)
        addVertex(vec3(-hx, -hy, -hz), vec3(0, -1, 0), vec2(0, 0), col);
        addVertex(vec3(hx, -hy, -hz), vec3(0, -1, 0), vec2(1, 0), col);
        addVertex(vec3(hx, -hy, hz), vec3(0, -1, 0), vec2(1, 1), col);
        addVertex(vec3(-hx, -hy, hz), vec3(0, -1, 0), vec2(0, 1), col);

        // Top face (+Y)
        addVertex(vec3(-hx, hy, hz), vec3(0, 1, 0), vec2(0, 0), col);
        addVertex(vec3(hx, hy, hz), vec3(0, 1, 0), vec2(1, 0), col);
        addVertex(vec3(hx, hy, -hz), vec3(0, 1, 0), vec2(1, 1), col);
        addVertex(vec3(-hx, hy, -hz), vec3(0, 1, 0), vec2(0, 1), col);

        // Indices (two triangles per face)
        for (uint32_t i = 0; i < 6; ++i)
        {
            uint32_t base = i * 4;
            addTriangle(base + 0, base + 1, base + 2);
            addTriangle(base + 0, base + 2, base + 3);
        }
    }
};