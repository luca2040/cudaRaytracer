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

__device__ __forceinline__ float3_L make_float3_L(float x, float y, float z) noexcept
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

__host__ __device__ __forceinline__ uchar4 make_uchar4_from_int(const int c)
{
  uchar4 retVal;

  retVal.x = (c >> 16) & 0xFF; // red
  retVal.y = (c >> 8) & 0xFF;  // green
  retVal.z = c & 0xFF;         // blue
  retVal.w = 255;              // alpha

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

// t -> 0 = color1 - t -> 1 = color2
__device__ __forceinline__ int colorMix(int &color1, int &color2, float &t) noexcept
{
  int r1 = (color1 >> 16) & 0xFF;
  int g1 = (color1 >> 8) & 0xFF;
  int b1 = color1 & 0xFF;

  int r2 = (color2 >> 16) & 0xFF;
  int g2 = (color2 >> 8) & 0xFF;
  int b2 = color2 & 0xFF;

  float t2 = 1 - t;

  int r = static_cast<int>(t2 * r1 + t * r2);
  int g = static_cast<int>(t2 * g1 + t * g2);
  int b = static_cast<int>(t2 * b1 + t * b2);

  return (r << 16) | (g << 8) | b;
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