#pragma once

#include "../definitions/RenderingStructs.cuh"
#include "ClosestHitKernel.cuh"
#include "MissingHitKernel.cuh"

__device__ __forceinline__ void traceRay(SceneMemoryPointers memPointers,
                                         ray &ray, RayData &rayData,
                                         const float3_L bgColor)
{
  // Bruteforce all the triangles - TODO: optimize this shit

  for (size_t depth = 0; depth < RAY_MAX_REFLECTIONS; depth++)
  {

    float currentZbuf = INFINITY;
    triangleidx hitTriangle;
    float3_L hitPos;

    for (size_t i = 0; i < memPointers.triangleNum; i++)
    {
      triangleidx triangle = memPointers.triangles[i];

      float t, u, v;
      float3_L rayHit;

      const float3_L *pointarray = memPointers.pointarray;

      bool hasIntersected = rayTriangleIntersection(ray,
                                                    pointarray[triangle.v1], pointarray[triangle.v2], pointarray[triangle.v3],
                                                    t, u, v,
                                                    rayHit);

      if (hasIntersected && (t < currentZbuf))
      {
        currentZbuf = t;
        hitTriangle = triangle;
        hitPos = rayHit;
      }
    }

    if (currentZbuf == INFINITY)
    {
      onHitMissing(memPointers,
                   ray, rayData,
                   bgColor);
      break;
    }

    bool stopReflection = onClosestHit(memPointers,
                                       ray, rayData, hitTriangle, hitPos,
                                       bgColor);
    if (stopReflection)
      break;
  }
}