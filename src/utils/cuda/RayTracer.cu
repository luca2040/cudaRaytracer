#include <iostream>
#include <cstdint>

#include "RayTracer.cuh"
#include "../DrawValues.h"
#include "../../scene/structs/Scene.h"
#include "../../scene/structs/SceneObject.h"
#include "../../math/cuda/CudaMath.cuh"

#include "kernels/RaytraceKernel.cuh"

#include "../../third_party/tracy/tracy/Tracy.hpp"

// Device Scene
Scene *d_scene;
size_t sceneSize;

void cudaAllocateScene()
{
  // Allocate all the memory needed
  cudaMalloc(&scene.d_pointarray, scene.pointsSize);
  cudaMalloc(&scene.d_triangles, scene.triangleSize);
  cudaMalloc(&scene.d_sceneobjects, scene.sceneObjectsSize);

  sceneSize = sizeof(Scene);
  cudaMalloc(&d_scene, sceneSize);

  // Copy the triangles index array, since its always static - not anymore because normals need to be recalculated each frame
  // cudaMemcpy(d_triangles, triangles, triangleSize, cudaMemcpyHostToDevice);
}

void cudaCleanup()
{
  cudaFree(scene.d_pointarray);
  cudaFree(scene.d_triangles);
  cudaFree(scene.d_sceneobjects);

  cudaFree(d_scene);
}

void rayTrace(
    uchar4 *pixelBuffer,
    const int bgColor)
{
  ZoneScopedN("rayTrace function");

  // cudaMemcpy(d_pixelBuffer, pixelBuffer, pixelBufferSize, cudaMemcpyHostToDevice);
  cudaMemcpy(scene.d_pointarray, scene.transformedPoints, scene.pointsSize, cudaMemcpyHostToDevice);
  cudaMemcpy(scene.d_triangles, scene.triangles, scene.triangleSize, cudaMemcpyHostToDevice);
  cudaMemcpy(scene.d_sceneobjects, scene.sceneobjects, scene.sceneObjectsSize, cudaMemcpyHostToDevice);

  cudaMemcpy(d_scene, &scene, sceneSize, cudaMemcpyHostToDevice);

  constexpr dim3 blockDim(16, 16);
  constexpr dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

  float3_L f3lBg = intColToF3l(bgColor);

  rayTraceKernel<<<gridDim, blockDim>>>(
      pixelBuffer,
      d_scene,
      WIDTH, HEIGHT,
      f3lBg);
}