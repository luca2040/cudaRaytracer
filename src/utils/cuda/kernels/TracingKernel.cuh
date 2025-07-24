#pragma once

#include "../definitions/RenderingStructs.cuh"
#include "ClosestHitKernel.cuh"
#include "MissingHitKernel.cuh"
#include "debug/IndexToUniqueColor.cuh"

__device__ __forceinline__ void traceRay(Scene *scene,
                                         ray &ray, RayData &rayData,
                                         const float3_L bgColor)
{
  // Bruteforce all the triangles - TODO: optimize this shit

  for (size_t depth = 0; depth < RAY_MAX_REFLECTIONS; depth++)
  {

    float currentZbuf = INFINITY;
    triangleidx hitTriangle;
    float3_L hitPos;

    for (size_t objNum = 0; objNum < scene->sceneobjectsNum; objNum++)
    {
      SceneObject currentObj = scene->d_sceneobjects[objNum];
      bool rayIntersectsObj = rayBoxIntersection(ray, currentObj.boundingBox);

      if (rayIntersectsObj)
      {
        if (scene->boundingBoxDebugView)
        {
          rayData.color = indexToColor(objNum, scene->sceneobjectsNum);
          break;
        }

        for (size_t i = currentObj.triangleStartIdx; i < currentObj.triangleEndIdx; i++)
        {
          triangleidx triangle = scene->d_triangles[i];

          float t, u, v;
          float3_L rayHit;

          const float3_L *pointarray = scene->d_pointarray;

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
      }
    }

    if (currentZbuf == INFINITY)
    {
      onHitMissing(scene,
                   ray, rayData,
                   bgColor);
      break;
    }

    bool stopReflection = onClosestHit(scene,
                                       ray, rayData, hitTriangle, hitPos,
                                       bgColor);
    if (stopReflection)
      break;
  }
}