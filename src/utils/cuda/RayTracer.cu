#include <iostream>
#include <cstdint>

#include "RayTracer.cuh"
#include "../DrawValues.h"
#include "../../scene/structs/Scene.h"
#include "../../scene/structs/SceneObject.h"
#include "../../math/cuda/CudaMath.cuh"

#include "kernels/RaytraceKernel.cuh"

#include "../../third_party/tracy/tracy/Tracy.hpp"
#include "../../third_party/tracy/tracy/TracyC.h"

void cudaAllocateScene()
{
  sceneStructSize = sizeof(Scene);

  // Allocate all the memory needed - * means cudamemcpy'd each frame
  cudaMalloc(&scene->d_pointarray, scene->pointsSize);
  cudaMalloc(&scene->d_trsfrmdpoints, scene->pointsSize);
  cudaMalloc(&scene->d_pointToObjIdxTable, scene->pointTableSize);
  cudaMalloc(&scene->d_triangles, scene->triangleSize);
  cudaMalloc(&scene->d_sceneobjects, scene->sceneObjectsSize);
  cudaMalloc(&scene->d_transformMatrices, scene->matricesSize); // *
  cudaMalloc(&d_scene, sceneStructSize);                        // *

  // Initial cudamemcpy
  cudaMemcpy(scene->d_pointarray, scene->points, scene->pointsSize, cudaMemcpyHostToDevice);
  cudaMemcpy(scene->d_triangles, scene->triangles, scene->triangleSize, cudaMemcpyHostToDevice);
  cudaMemcpy(scene->d_pointToObjIdxTable, scene->pointToObjIdxTable, scene->pointTableSize, cudaMemcpyHostToDevice);
  cudaMemcpy(scene->d_sceneobjects, scene->sceneobjects, scene->sceneObjectsSize, cudaMemcpyHostToDevice);
  cudaMemcpy(scene->d_transformMatrices, scene->transformMatrices, scene->matricesSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scene, scene, sceneStructSize, cudaMemcpyHostToDevice);
}

void cudaCleanup()
{
  cudaFreeHost(scene->transformMatrices);

  cudaFree(scene->d_pointarray);
  cudaFree(scene->d_trsfrmdpoints);
  cudaFree(scene->d_triangles);
  cudaFree(scene->d_sceneobjects);
  cudaFree(scene->d_transformMatrices);
  cudaFree(d_scene);
}

void rayTrace(
    uchar4 *pixelBuffer,
    const int bgColor)
{
  ZoneScopedN("rayTrace function");
  TracyCZoneN(cudaTrace, "Cuda trace", true);

  constexpr dim3 blockDim(RAYTRACE_BLOCK_SIDE, RAYTRACE_BLOCK_SIDE);
  constexpr dim3 gridDim((WIDTH + (RAYTRACE_BLOCK_SIDE - 1)) / RAYTRACE_BLOCK_SIDE,
                         (HEIGHT + (RAYTRACE_BLOCK_SIDE - 1)) / RAYTRACE_BLOCK_SIDE);

  float3_L f3lBg = intColToF3l(bgColor);

  rayTraceKernel<<<gridDim, blockDim>>>(
      pixelBuffer,
      d_scene,
      WIDTH, HEIGHT,
      f3lBg);

  TracyCZoneEnd(cudaTrace);
}