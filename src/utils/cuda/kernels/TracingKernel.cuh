#pragma once

#include "../definitions/RenderingStructs.cuh"
#include "ClosestHitKernel.cuh"
#include "MissingHitKernel.cuh"
#include "FinalizeKernel.cuh"
#include "debug/IndexToUniqueColor.cuh"
#include "../../../math/cuda/CudaGraphicsMath.cuh"

__device__ __forceinline__ void traceRay(Scene *scene, uint &RNGstate,
                                         Ray &currentRay, RayData &rayData)
{
  for (int depth = 0; depth < scene->maxRayReflections; depth++)
  {
    float currentZbuf = INFINITY;
    triangleidx hitTriangle;
    float3_L hitPos;

    Ray invertedDirRay = currentRay;
    invertedDirRay.direction = inverse(currentRay.direction);

    for (size_t objNum = 0; objNum < scene->sceneobjectsNum; objNum++)
    {
      SceneObject currentObj = scene->d_sceneobjects[objNum];

      // rayBoxIntersection NEEDS a ray with inversed direction
      bool rayDoesntIntersectObj = rayBoxIntersection(invertedDirRay, currentObj.boundingBox);

      if (rayDoesntIntersectObj)
        continue;

      if (scene->boundingBoxDebugView)
      {
        rayData.color = indexToColor(objNum, scene->sceneobjectsNum);
        break;
      }

      for (size_t i = currentObj.triangleStartIdx; i < currentObj.triangleStartIdx + currentObj.triangleNum; i++)
      {
        triangleidx triangle = scene->d_triangles[i];

        float t, u, v;
        float3_L rayHit;

        const float3_L *pointarray = scene->d_trsfrmdpoints;

        bool hasIntersected = rayTriangleIntersection(currentRay,
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

    if (currentZbuf == INFINITY)
    {
      onHitMissing(scene, RNGstate,
                   currentRay, rayData);
      break;
    }

    bool stopReflection = onClosestHit(scene, RNGstate,
                                       currentRay, rayData, hitTriangle, hitPos);
    if (stopReflection)
      break;
  }

  // Apply lights, or whatever
  finalizeColor(scene, RNGstate,
                currentRay, rayData);
}