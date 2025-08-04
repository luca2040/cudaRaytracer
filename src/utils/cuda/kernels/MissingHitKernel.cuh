#pragma once

#include "TracingKernel.cuh"

__device__ __forceinline__ void onHitMissing(Scene *scene, uint &RNGstate,
                                             Ray &ray, RayData &rayData)
{
  float horizonStart = -0.2f;
  float horizonEnd = 0.5f;

  float3_L groundColor = make_float3_L(0.4f) * scene->environmentLight;
  float3_L horizonColor = make_float3_L(1.0f) * scene->environmentLight;

  float3_L color; // This point in the sky's color
  float3_L light; // Same for the light

  float dotResult = dot3_cuda(ray.direction, make_float3_L(0.0f, -1.0f, 0.0f));

  if (dotResult < horizonStart)
  {
    color = groundColor;
    light = make_float3_L(0.0f);
  }
  else
  {
    float clampedDot = fminf(dotResult, horizonEnd); // Result goes from horizonStart to horizonEnd

    float skyPercent = (clampedDot - horizonStart) / (horizonEnd - horizonStart); // Map the last value from 0.0f to 1.0f
    float groundPercent = 1.0f - skyPercent;

    color = (scene->backgroundColor * skyPercent) + (horizonColor * groundPercent);
    light = color * scene->environmentLight; // More light from the top
  }

  if (!rayData.hasHit)
  {
    rayData.rayLight = color * scene->environmentLight;
    return; // If the ray goes directly into the sky just apply the color
  }

  rayData.rayLight += light * rayData.color;
}