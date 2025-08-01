#pragma once

#include "TracingKernel.cuh"

__device__ __forceinline__ void onHitMissing(Scene *scene,
                                             ray &ray, RayData &rayData)
{
  rayData.color = rayData.color + (scene->backgroundColor * rayData.rayLight);
  // rayData.color = make_float3_L(0.5f, 0, 0);
}