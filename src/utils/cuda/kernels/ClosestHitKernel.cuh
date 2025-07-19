#pragma once

#include "TracingKernel.cuh"

__device__ void traceRay(SceneMemoryPointers memPointers,
                         ray &ray, RayData &rayData,
                         const float3_L bgColor);

__device__ __forceinline__ bool onClosestHit(SceneMemoryPointers memPointers,
                                             ray &ray, RayData &rayData, triangleidx hitTriangle, float3_L hitPos,
                                             const float3_L bgColor)
{
  float reflectiveness = hitTriangle.reflectiveness;

  rayData.color = rayData.color + (intColToF3l(hitTriangle.col) * rayData.rayLight * (1.0f - reflectiveness));
  rayData.rayLight *= reflectiveness;

  if (reflectiveness < EPSILON)
    return true;

  ray.origin = hitPos;
  reflectRay(ray.direction, hitTriangle.normal);

  return false;
}