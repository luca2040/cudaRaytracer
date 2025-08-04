#pragma once

#include <cmath>

struct int2_L
{
  int x;
  int y;

  inline int2_L() = default;
  inline int2_L(int x, int y) : x(x), y(y) {}
};

struct float2_L
{
  float x;
  float y;

  inline float2_L() = default;
  inline float2_L(float x, float y) : x(x), y(y) {}

  inline float2_L &operator+=(const float2_L &rhs) noexcept
  {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }

  inline float2_L &operator-=(const float2_L &rhs) noexcept
  {
    x -= rhs.x;
    y -= rhs.y;
    return *this;
  }
};

struct float3_L
{
  float x;
  float y;
  float z;

  inline float3_L() = default;
  inline float3_L(float x, float y, float z) : x(x), y(y), z(z) {}

#ifndef __CUDACC__
  bool operator==(const float3_L &) const = default;
#endif

  inline float3_L &operator+=(const float3_L &rhs) noexcept
  {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
  }

  inline float3_L &operator-=(const float3_L &rhs) noexcept
  {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
  }

  inline operator float2_L() const
  {
    return float2_L{x, y};
  }
};

struct float4_L
{
  float x;
  float y;
  float z;
  float k;

  inline float4_L() = default;
  inline float4_L(float x, float y, float z, float k) : x(x), y(y), z(z), k(k) {}
};

struct mat3x3
{
  float3_L rows[3];

  inline const float3_L &operator[](int i) const { return rows[i]; }
  inline float3_L &operator[](int i) { return rows[i]; }

  inline float3_L operator*(const float3_L &v) const
  {
    return {
        rows[0].x * v.x + rows[0].y * v.y + rows[0].z * v.z,
        rows[1].x * v.x + rows[1].y * v.y + rows[1].z * v.z,
        rows[2].x * v.x + rows[2].y * v.y + rows[2].z * v.z};
  }

  inline mat3x3 operator*(const mat3x3 &b) const
  {
    mat3x3 r;
    for (int i = 0; i < 3; ++i)
    {
      r.rows[i] = {
          rows[i].x * b[0].x + rows[i].y * b[1].x + rows[i].z * b[2].x,
          rows[i].x * b[0].y + rows[i].y * b[1].y + rows[i].z * b[2].y,
          rows[i].x * b[0].z + rows[i].y * b[1].z + rows[i].z * b[2].z};
    }
    return r;
  }
};

struct mat4x4
{
  float4_L rows[4];

  inline const float4_L &operator[](int i) const { return rows[i]; }
  inline float4_L &operator[](int i) { return rows[i]; }

  inline mat4x4 operator*(const mat4x4 &b) const
  {
    mat4x4 r;
    for (int i = 0; i < 4; ++i)
    {
      r.rows[i] = {
          rows[i].x * b[0].x + rows[i].y * b[1].x + rows[i].z * b[2].x + rows[i].k * b[3].x,
          rows[i].x * b[0].y + rows[i].y * b[1].y + rows[i].z * b[2].y + rows[i].k * b[3].y,
          rows[i].x * b[0].z + rows[i].y * b[1].z + rows[i].z * b[2].z + rows[i].k * b[3].z,
          rows[i].x * b[0].k + rows[i].y * b[1].k + rows[i].z * b[2].k + rows[i].k * b[3].k};
    }
    return r;
  }
};

#define INT_TO_FLOAT3_COLOR(intColor, floatColor)                      \
  floatColor.x = static_cast<float>((intColor >> 16) & 0xFF) / 255.0f; \
  floatColor.y = static_cast<float>((intColor >> 8) & 0xFF) / 255.0f;  \
  floatColor.z = static_cast<float>(intColor & 0xFF) / 255.0f

struct Material
{
  float3_L col;
  float diffuse; // When a diffuse ray is sent, how much it affects the color.

  Material() = default;
  Material(int matCol, float diffuse)
      : diffuse(diffuse)
  {
    INT_TO_FLOAT3_COLOR(matCol, col);
  }

#ifndef __CUDACC__
  bool operator==(const Material &) const = default;
#endif
};

// Indicates a triangle based on its vertices indexes in the vert array
struct triangleidx
{
  size_t v1;
  size_t v2;
  size_t v3;

  size_t materialIdx;

  float3_L normal;

  inline triangleidx() = default;
  inline triangleidx(size_t v1, size_t v2, size_t v3, size_t materialIdx)
      : v1(v1), v2(v2), v3(v3), materialIdx(materialIdx) {}
};