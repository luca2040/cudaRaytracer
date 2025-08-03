#pragma once

#include "TracingKernel.cuh"

__device__ __forceinline__ void onHitMissing(Scene *scene,
                                             Ray &ray, RayData &rayData)
{
  float horizonStart = -0.6f;
  float horizonEnd = -0.2f;

  float3_L groundColor = make_float3_L(0.4f, 0.4f, 0.4f);

  float dotResult = dot3_cuda(ray.direction, make_float3_L(0.0f, -1.0f, 0.0f));
  float clampedDot = fminf(fmaxf(dotResult, horizonStart), horizonEnd); // Result goes from horizonStart to horizonEnd

  float skyPercent = (clampedDot - horizonStart) / (horizonEnd - horizonStart); // Map the last value from 0.0f to 1.0f
  float groundPercent = 1.0f - skyPercent;

  float3_L color = (scene->backgroundColor * skyPercent) + (groundColor * groundPercent);

  rayData.rayLight = rayData.rayLight + color; // Always full light from environment
  rayData.color = rayData.color + (color * rayData.reflReduction);
}