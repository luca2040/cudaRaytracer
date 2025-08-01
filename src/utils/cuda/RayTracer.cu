#include <iostream>
#include <cstdint>

#include "RayTracer.cuh"
#include "../DrawValues.h"
#include "../../scene/structs/Scene.h"
#include "../../scene/structs/SceneObject.h"
#include "../../math/cuda/CudaMath.cuh"

#include "kernels/RaytraceKernel.cuh"

#include "../Profiling.h"

void cudaAllocateScene()
{
  sceneStructSize = sizeof(Scene);

  // Allocate all the memory needed - * means cudamemcpy'd each frame
  cudaMalloc(&scene->d_pointarray, scene->pointsSize);
  cudaMalloc(&scene->d_trsfrmdpoints, scene->pointsSize);
  cudaMalloc(&scene->d_pointToObjIdxTable, scene->pointTableSize);
  cudaMalloc(&scene->d_triangles, scene->triangleSize);
  cudaMalloc(&scene->d_sceneobjects, scene->sceneObjectsSize);
  cudaMalloc(&d_scene, sceneStructSize); // *

  // Initial cudamemcpy
  cudaMemcpy(scene->d_pointarray, scene->points, scene->pointsSize, cudaMemcpyHostToDevice);
  cudaMemcpy(scene->d_triangles, scene->triangles, scene->triangleSize, cudaMemcpyHostToDevice);
  cudaMemcpy(scene->d_pointToObjIdxTable, scene->pointToObjIdxTable, scene->pointTableSize, cudaMemcpyHostToDevice);
  cudaMemcpy(scene->d_sceneobjects, scene->sceneobjects, scene->sceneObjectsSize, cudaMemcpyHostToDevice);
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

  cudaFreeHost(scene);
}

void rayTrace(uchar4 *pixelBuffer, int renderWidth, int renderHeight)
{
  ZONESCOPEDNC("rayTrace function", PROFILER_LIME_GREEN);
  TRACYCZONENC(cudaTrace, "Cuda trace", true, PROFILER_GOLD);

  dim3 blockDim(RAYTRACE_BLOCK_SIDE, RAYTRACE_BLOCK_SIDE);
  dim3 gridDim((renderWidth + (RAYTRACE_BLOCK_SIDE - 1)) / RAYTRACE_BLOCK_SIDE,
               (renderHeight + (RAYTRACE_BLOCK_SIDE - 1)) / RAYTRACE_BLOCK_SIDE);

  rayTraceKernel<<<gridDim, blockDim>>>(
      pixelBuffer,
      d_scene,
      renderWidth, renderHeight);

  if (scene->afterTraceSync)
    cudaDeviceSynchronize();

  TRACYCZONEEND(cudaTrace);
}