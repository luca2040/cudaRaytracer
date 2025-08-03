#pragma once

#include "TracingKernel.cuh"

__device__ __forceinline__ void finalizeColor(Scene *scene, uint &RNGstate,
                                              Ray &ray, RayData &rayData)
{
  // rayData.color = min(rayData.rayLight, make_float3_L(1.0f, 1.0f, 1.0f)); // Apply color as light
  rayData.color = min(rayData.color, make_float3_L(1.0f, 1.0f, 1.0f));
}