#include <iostream>
#include <cstdint>

#include "RayTracer.cuh"
#include "../DrawValues.h"
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

    const float3_L camPos,
    const float3_L camViewOrigin,
    const float3_L imageX,
    const float3_L imageY,
    float inverseWidthMinus,
    float inverseHeightMinus,

    const float3_L *pointarray,
    const triangleidx *triangles,
    const SceneObject *sceneobjects,
    size_t sceneobjectsNum,

    size_t pointarraySize,
    size_t trianglesSize,
    size_t sceneobjectsSize,

    const int bgColor)
{
  ZoneScopedN("rayTrace function");

  // cudaMemcpy(d_pixelBuffer, pixelBuffer, pixelBufferSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pointarray, pointarray, pointarraySize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_triangles, triangles, trianglesSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sceneobjects, sceneobjects, sceneobjectsSize, cudaMemcpyHostToDevice);

  constexpr dim3 blockDim(16, 16);
  constexpr dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

  float3_L f3lBg = intColToF3l(bgColor);
  SceneMemoryPointers memPointers = SceneMemoryPointers(d_pointarray, d_triangles, d_sceneobjects,
                                                        sceneobjectsNum);

  rayTraceKernel<<<gridDim, blockDim>>>(
      pixelBuffer,
      camPos, camViewOrigin, imageX, imageY,
      inverseWidthMinus, inverseHeightMinus,
      memPointers,
      WIDTH, HEIGHT,
      f3lBg);
}