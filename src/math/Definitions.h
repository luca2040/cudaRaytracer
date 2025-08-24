#pragma once

#include <cmath>

#ifdef __CUDACC__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define HOST_DEVICE_INLINE inline
#endif

struct int2_L
{
  int x;
  int y;

  HOST_DEVICE_INLINE int2_L() = default;
  HOST_DEVICE_INLINE int2_L(int x, int y) : x(x), y(y) {}
};

struct float2_L
{
  float x;
  float y;

  HOST_DEVICE_INLINE float2_L() = default;
  HOST_DEVICE_INLINE float2_L(float x, float y) : x(x), y(y) {}

  HOST_DEVICE_INLINE float2_L &operator+=(const float2_L &rhs) noexcept
  {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }

  HOST_DEVICE_INLINE float2_L &operator-=(const float2_L &rhs) noexcept
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

  HOST_DEVICE_INLINE float3_L() = default;
  HOST_DEVICE_INLINE float3_L(float x, float y, float z) : x(x), y(y), z(z) {}

#ifndef __CUDACC__
  bool operator==(const float3_L &) const = default;
#endif

  HOST_DEVICE_INLINE float3_L &operator+=(const float3_L &rhs) noexcept
  {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
  }

  HOST_DEVICE_INLINE float3_L &operator-=(const float3_L &rhs) noexcept
  {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
  }

  HOST_DEVICE_INLINE operator float2_L() const
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

  HOST_DEVICE_INLINE float4_L() = default;
  HOST_DEVICE_INLINE float4_L(float x, float y, float z, float k) : x(x), y(y), z(z), k(k) {}
};

struct mat3x3
{
  float3_L rows[3];

  HOST_DEVICE_INLINE const float3_L &operator[](int i) const { return rows[i]; }
  HOST_DEVICE_INLINE float3_L &operator[](int i) { return rows[i]; }

  HOST_DEVICE_INLINE float3_L operator*(const float3_L &v) const
  {
    return {
        rows[0].x * v.x + rows[0].y * v.y + rows[0].z * v.z,
        rows[1].x * v.x + rows[1].y * v.y + rows[1].z * v.z,
        rows[2].x * v.x + rows[2].y * v.y + rows[2].z * v.z};
  }

  HOST_DEVICE_INLINE mat3x3 operator*(const mat3x3 &b) const
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

  HOST_DEVICE_INLINE const float4_L &operator[](int i) const { return rows[i]; }
  HOST_DEVICE_INLINE float4_L &operator[](int i) { return rows[i]; }

  HOST_DEVICE_INLINE mat4x4 operator*(const mat4x4 &b) const
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
  float3_L col;         // Ray color
  float diffuse;        // When a diffuse ray is sent, how much it affects the color.
  float3_L emCol;       // Emission color
  float emStren;        // Emission strenght
  float reflectiveness; // Reflectiveness value. 1.0f means full mirror

  Material() = default;
  Material(int matCol, float diffuse, int emColor, float emStren, float reflectiveness)
      : diffuse(diffuse), emStren(emStren), reflectiveness(reflectiveness)
  {
    INT_TO_FLOAT3_COLOR(matCol, col);
    INT_TO_FLOAT3_COLOR(emColor, emCol);
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

  HOST_DEVICE_INLINE triangleidx() = default;
  HOST_DEVICE_INLINE triangleidx(size_t v1, size_t v2, size_t v3, size_t materialIdx)
      : v1(v1), v2(v2), v3(v3), materialIdx(materialIdx) {}
};