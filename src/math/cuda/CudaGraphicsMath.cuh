#pragma once

#include "CudaRNG.cuh"
#include "../../scene/structs/Sphere.h"

// Möller–Trumbore intersection algorithm for ray-triangle intersection
// Returns true if intersection occurs, false otherwise
// t: distance along ray, u/v: barycentric coordinates, outHit: intersection point
__device__ __forceinline__ bool rayTriangleIntersection(
    const Ray &r,
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

// RETURNS TRUE WHEN RAY DOESNT INTERSECT THE BOX
// NEEDS RAY WITH INVERTED DIRECTION (for optimizations in ray check loop)
__device__ __forceinline__ bool rayBoxIntersection(Ray testRay, AABB bb)
{
  float3_L tLowComponents = (bb.l - testRay.origin) * testRay.direction;
  float3_L tHighComponents = (bb.h - testRay.origin) * testRay.direction;

  float3_L tCloseComponents = min(tLowComponents, tHighComponents);
  float3_L tFarComponents = max(tLowComponents, tHighComponents);

  float tClose = fmaxf(fmaxf(tCloseComponents.x, tCloseComponents.y), tCloseComponents.z);
  float tFar = fminf(fminf(tFarComponents.x, tFarComponents.y), tFarComponents.z);

  // If tClose is less than 0 it intersects behind the ray, and that no good.
  // Intersection happens if tClose <= tFar and neither of them is NaN.
  // return !(tClose < 0 ? false : tClose <= tFar);
  // return !(tClose <= tFar && tFar >= 0.0f);
  return tClose > tFar || tFar < 0.0f;
}

__device__ __forceinline__ bool raySphereIntersection(Ray testRay, Sphere sphere,
                                                      float &t0, float3_L &hitPoint, float3_L &hitNormal)
{
  float3_L l = sphere.center - testRay.origin;
  float tcenter = dot3_cuda(l, testRay.direction);

  float vc = sqrtf(dot3_cuda(l, l)); // distance from vector start to center of sphere
  bool inside = vc < sphere.radius;

  if (tcenter < 0 && !inside) // Wrong direction, if it even collides
    return false;

  float d = sqrtf(dot3_cuda(l, l) - (tcenter * tcenter));

  if (d > sphere.radius) // Ray doesnt intersect
    return false;

  float tinside = sqrtf((sphere.radius * sphere.radius) - (d * d));
  t0 = inside ? tcenter + tinside : tcenter - tinside; // actually use t1 if inside

  if (t0 < EPSILON)
    return false;

  hitPoint = testRay.origin + testRay.direction * t0;
  hitNormal = normalize3_cuda(hitPoint - sphere.center);
  return true;
}

__device__ __forceinline__ void reflectRay(float3_L &rayDir, float3_L &normal)
{
  float d = dot3_cuda(rayDir, normal);

  if (d > 0.0f)
  {
    normal = normal * -1.0f;
    d = -d;
  }

  rayDir = rayDir - normal * (2.0f * d);
}

__device__ __forceinline__ void randomSemisphereVector(float3_L &rayDir, float3_L &normal, uint &RNGstate)
{
  // Check if normal is reversed
  float d = dot3_cuda(rayDir, normal);
  if (d > 0.0f)
    normal = normal * -1.0f;

  // Generate ray on semisphere
  float3_L ranVec = randomVector(RNGstate);
  float randDot = dot3_cuda(ranVec, normal);
  if (randDot < 0.0f)
    ranVec = ranVec * -1.0f;

  rayDir = ranVec;
}

__device__ __forceinline__ void lambertianVector(float3_L &rayDir, float3_L &normal, uint &RNGstate)
{
  // Check if normal is reversed
  float d = dot3_cuda(rayDir, normal);
  if (d > 0.0f)
    normal = normal * -1.0f;

  // Generate ray on semisphere
  float3_L ranVec = randomVector(RNGstate);
  float randDot = dot3_cuda(ranVec, normal);
  if (randDot < 0.0f)
    ranVec = ranVec * -1.0f;

  rayDir = normalize3_cuda(ranVec + normal);
}

__device__ __forceinline__ void refractRay(float3_L &rayDir, float3_L &normal)
{
  // Check if normal is reversed
  float d = dot3_cuda(rayDir, normal);
  if (d > 0.0f)
    normal = normal * -1.0f;

  rayDir = rayDir - normal * (2.0f * d);
}