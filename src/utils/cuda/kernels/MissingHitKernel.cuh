#pragma once

#include "TracingKernel.cuh"

__device__ __forceinline__ void onHitMissing(SceneMemoryPointers memPointers,
                                             ray &ray, RayData &rayData,
                                             const float3_L bgColor)
{
  rayData.color = rayData.color + (bgColor * rayData.rayLight);
  // rayData.color = make_float3_L(0.5f, 0, 0);
}