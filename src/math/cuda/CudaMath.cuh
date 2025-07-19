#pragma once

#include "../Definitions.h"
#include "../../utils/DrawValues.h"

struct ray
{
  float3_L origin;
  float3_L direction; // Normalized direction vector

  __host__ __device__ __forceinline__
  ray() = default;

  __host__ __device__ __forceinline__
  ray(float3_L origin, float3_L direction) : origin(origin), direction(direction) {}
};

__host__ __device__ __forceinline__ static float3_L make_float3_L(float x, float y, float z) noexcept
{
  float3_L v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
};

__device__ __forceinline__ float3_L operator+(const float3_L &a, const float3_L &b) noexcept
{
  return make_float3_L(a.x + b.x, a.y + b.y, a.z + b.z);
};

__device__ __forceinline__ float3_L operator-(const float3_L &a, const float3_L &b) noexcept
{
  return make_float3_L(a.x - b.x, a.y - b.y, a.z - b.z);
};

__device__ __forceinline__ float3_L operator*(const float3_L &a, const float &b) noexcept
{
  return make_float3_L(a.x * b, a.y * b, a.z * b);
};

__device__ __forceinline__ float dot3_cuda(const float3_L &a, const float3_L &b) noexcept
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3_L cross3_cuda(const float3_L &a, const float3_L &b) noexcept
{
  return make_float3_L(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x);
}

__host__ __device__ __forceinline__ float3_L intColToF3l(const int c)
{
  float x = static_cast<float>((c >> 16) & 0xFF) / 255.0f; // red
  float y = static_cast<float>((c >> 8) & 0xFF) / 255.0f;  // green
  float z = static_cast<float>(c & 0xFF) / 255.0f;         // blue

  return make_float3_L(x, y, z);
}

__host__ __device__ __forceinline__ uchar4 make_uchar4_from_f3l(const float3_L c)
{
  uchar4 retVal;

  retVal.x = static_cast<char>(min(c.x, 1.0f) * 255.0f); // red
  retVal.y = static_cast<char>(min(c.y, 1.0f) * 255.0f); // green
  retVal.z = static_cast<char>(min(c.z, 1.0f) * 255.0f); // blue
  retVal.w = 255;                                        // alpha

  return retVal;
}

__device__ __forceinline__ float3_L normalize3_cuda(const float3_L &a) noexcept
{
  return a * rsqrtf(dot3_cuda(a, a));
}

// Möller–Trumbore intersection algorithm for ray-triangle intersection
// Returns true if intersection occurs, false otherwise
// t: distance along ray, u/v: barycentric coordinates, outHit: intersection point
__device__ __forceinline__ bool rayTriangleIntersection(
    const ray &r,
    const float3_L &v0,
    const float3_L &v1,
    const float3_L &v2,
    float &t,
    float &u,
    float &v,
    float3_L &outHit)
{
  float3_L edge1 = v1 - v0;
  float3_L edge2 = v2 - v0;
  float3_L ray_cross_e2 = cross3_cuda(r.direction, edge2);
  float det = dot3_cuda(edge1, ray_cross_e2);

  if (det > -EPSILON && det < EPSILON)
    return false; // This ray is parallel to this triangle.

  float inv_det = 1.0 / det;
  float3_L s = r.origin - v0;
  u = inv_det * dot3_cuda(s, ray_cross_e2);

  if ((u < 0 && abs(u) > EPSILON) || (u > 1 && abs(u - 1) > EPSILON))
    return false;

  float3_L s_cross_e1 = cross3_cuda(s, edge1);
  v = inv_det * dot3_cuda(r.direction, s_cross_e1);

  if ((v < 0 && abs(v) > EPSILON) || (u + v > 1 && abs(u + v - 1) > EPSILON))
    return false;

  // At this stage we can compute t to find out where the intersection point is on the line.
  t = inv_det * dot3_cuda(edge2, s_cross_e1);

  if (t > EPSILON) // ray intersection
  {
    outHit = r.origin + r.direction * t;
    return true;
  }
  else // This means that there is a line intersection but not a ray intersection.
    return false;
}

__device__ __forceinline__ void reflectRay(float3_L &rayDir, float3_L &normal) noexcept
{
  float d = dot3_cuda(rayDir, normal);

  if (d > 0.0f)
  {
    normal = normal * -1.0f;
    d = -d;
  }

  rayDir = rayDir - normal * (2.0f * d);
}