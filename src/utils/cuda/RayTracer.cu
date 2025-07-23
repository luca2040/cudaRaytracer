#include <iostream>
#include <cstdint>

#include "RayTracer.cuh"
#include "../DrawValues.h"
#include "../../scene/structs/Scene.h"
#include "../../scene/structs/SceneObject.h"
#include "../../math/cuda/CudaMath.cuh"

#include "kernels/RaytraceKernel.cuh"
#include "definitions/RenderingStructs.cuh"

#include "../../third_party/tracy/tracy/Tracy.hpp"

// ######## Device parameters ########

float3_L *d_pointarray;
triangleidx *d_triangles;
SceneObject *d_sceneobjects;

void cudaAllocateAndCopy(size_t pointsSize,
                         size_t triangleSize,
                         size_t sceneobjectsSize)
{
  // Allocate all the memory needed
  cudaMalloc(&d_pointarray, pointsSize);
  cudaMalloc(&d_triangles, triangleSize);
  cudaMalloc(&d_sceneobjects, sceneobjectsSize);

  // Copy the triangles index array, since its always static - not now because normals need to be recalculated each frame
  // cudaMemcpy(d_triangles, triangles, triangleSize, cudaMemcpyHostToDevice);
}

void cudaCleanup()
{
  cudaFree(d_pointarray);
  cudaFree(d_triangles);
  cudaFree(d_sceneobjects);
}

void rayTrace(
    uchar4 *pixelBuffer,
    const int bgColor)
{
  ZoneScopedN("rayTrace function");

  // cudaMemcpy(d_pixelBuffer, pixelBuffer, pixelBufferSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pointarray, scene.transformedPoints, scene.pointsSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_triangles, scene.triangles, scene.triangleSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sceneobjects, scene.sceneobjects, scene.sceneObjectsSize, cudaMemcpyHostToDevice);

  constexpr dim3 blockDim(16, 16);
  constexpr dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

  float3_L f3lBg = intColToF3l(bgColor);
  SceneMemoryPointers memPointers = SceneMemoryPointers(d_pointarray, d_triangles, d_sceneobjects,
                                                        scene.sceneobjectsNum);

  auto &cam = scene.cam;

  rayTraceKernel<<<gridDim, blockDim>>>(
      pixelBuffer,
      cam.camPos, cam.camViewOrigin, cam.imageX, cam.imageY,
      cam.inverseWidthMinus, cam.inverseHeightMinus,
      memPointers,
      WIDTH, HEIGHT,
      f3lBg);
}