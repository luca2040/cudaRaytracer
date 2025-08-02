#pragma once

#include "TracingKernel.cuh"

__device__ __forceinline__ bool onClosestHit(Scene *scene,
                                             Ray &ray, RayData &rayData,
                                             triangleidx hitTriangle, float3_L hitPos)
{
  Material currentMaterial = scene->d_materials[hitTriangle.materialIdx];

  rayData.color = rayData.color + (intColToF3l(currentMaterial.col) * rayData.rayLight * (1.0f - currentMaterial.reflectiveness));
  rayData.rayLight *= currentMaterial.reflectiveness;

  if (currentMaterial.reflectiveness < EPSILON)
    return true;

  ray.origin = hitPos;
  reflectRay(ray.direction, hitTriangle.normal);

  return false;
}