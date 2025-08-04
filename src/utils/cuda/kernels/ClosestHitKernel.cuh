#pragma once

#include "TracingKernel.cuh"
#include "../../../math/cuda/CudaGraphicsMath.cuh"

// Return true if there is no reflection
__device__ __forceinline__ bool onClosestHit(Scene *scene, uint &RNGstate,
                                             Ray &ray, RayData &rayData,
                                             triangleidx hitTriangle, float3_L hitPos)
{
  Material mat = scene->d_materials[hitTriangle.materialIdx];
  if (!rayData.hasHit) // Indicates the ray has intersected at least a triangle
    rayData.hasHit = true;

  // Not used in actual rendering, just for the simplified one
  if (scene->simpleRender)
  {
    rayData.color = mat.col;
    return true;
  }

  // lastDiffuse -> 1 => col = mat.col | lastDiffuse -> 0 => col = rayData.color
  float3_L diffusedColor = mat.col * rayData.lastDiffuse + rayData.color * (1.0f - rayData.lastDiffuse);
  float3_L emittedLight = mat.emCol * mat.emStren;

  rayData.rayLight += emittedLight * rayData.color;
  rayData.color *= diffusedColor;

  ray.origin = hitPos;
  lambertianVector(ray.direction, hitTriangle.normal, RNGstate);

  rayData.lastDiffuse *= mat.diffuse;

  return false;
}