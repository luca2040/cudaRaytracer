#pragma once

#include "TracingKernel.cuh"

__device__ __forceinline__ void finalizeColor(Scene *scene, uint &RNGstate,
                                              Ray &ray, RayData &rayData)
{
  if (scene->simpleRender || scene->boundingBoxDebugView)
    return; // Dont apply lights in simple render mode

  rayData.color = rayData.rayLight;
}