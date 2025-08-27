#pragma once

#include "TracingKernel.cuh"
#include "../../../math/cuda/CudaGraphicsMath.cuh"

// Return true if there is no reflection
__device__ __forceinline__ bool onClosestHit(Scene *scene, uint &RNGstate,
                                             Ray &ray, RayData &rayData,
                                             triangleidx hitTriangle, float3_L hitPos)
{
  Material mat = scene->d_materials[hitTriangle.materialIdx];

  // Not used in actual rendering, just for the simplified one
  if (scene->simpleRender)
  {
    rayData.color = mat.col;
    return true;
  }

  // Dont compute both reflection and reflaction at the same time
  // Since this thing is kinda iterative just get the probability and it will produce the same result cuz of the exposition time
  bool doReflect = randomValue(RNGstate) < mat.reflectiveness;

  // lastDiffuse -> 1 => col = mat.col | lastDiffuse -> 0 => col = rayData.color
  float3_L diffusedColor = mat.col * rayData.lastDiffuse + rayData.color * (1.0f - rayData.lastDiffuse);
  float3_L emittedLight = mat.emCol * mat.emStren;

  rayData.rayLight += emittedLight * rayData.color;

  ray.origin = hitPos;
  if (doReflect)
  {
    if (mat.isMetal)
      rayData.color *= diffusedColor;
    else
      rayData.color *= make_float3_L(mat.reflectiveness);

    reflectRay(ray.direction, hitTriangle.normal);
  }
  else
  {
    rayData.color *= diffusedColor;
    lambertianVector(ray.direction, hitTriangle.normal, RNGstate);
  }

  rayData.lastDiffuse *= mat.diffuse;

  return false;
}