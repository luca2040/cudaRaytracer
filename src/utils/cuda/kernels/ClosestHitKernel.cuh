#pragma once

#include "TracingKernel.cuh"

// Return true if there is no reflection
__device__ __forceinline__ bool onClosestHit(Scene *scene,
                                             Ray &ray, RayData &rayData,
                                             triangleidx hitTriangle, float3_L hitPos)
{
  Material mat = scene->d_materials[hitTriangle.materialIdx];

  // float3_L emittedLight = mat.emissionColor * mat.emissivity;
  // rayData.rayLight = rayData.rayLight + (emittedLight * rayData.color);

  // rayData.color = rayData.color + (mat.col * rayData.reflReduction * (1.0f - mat.reflectiveness));
  // rayData.reflReduction *= mat.reflectiveness;

  rayData.color = rayData.color + mat.col;

  return true;

  // if (mat.reflectiveness < EPSILON)
  //   return true; // Cancel reflection

  // ray.origin = hitPos;
  // reflectRay(ray.direction, hitTriangle.normal);

  // return false;
}