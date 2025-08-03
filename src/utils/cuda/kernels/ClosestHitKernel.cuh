#pragma once

#include "TracingKernel.cuh"
#include "../../../math/cuda/CudaGraphicsMath.cuh"

// Return true if there is no reflection
__device__ __forceinline__ bool onClosestHit(Scene *scene, uint &RNGstate,
                                             Ray &ray, RayData &rayData,
                                             triangleidx hitTriangle, float3_L hitPos)
{
  Material mat = scene->d_materials[hitTriangle.materialIdx];

  rayData.color = rayData.color + (mat.col * 0.5f);

  ray.origin = hitPos;
  randomSemisphereVector(ray.direction, hitTriangle.normal, RNGstate);

  return false;
}