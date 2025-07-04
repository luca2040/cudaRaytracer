#include <iostream>
#include <cstdint>

#include "RayTracer.cuh"
#include "../DrawValues.h"

#include "kernels/RaytraceKernel.cuh"

#include "../../third_party/tracy/tracy/Tracy.hpp"

// ######## Device parameters ########

float3_L *d_pointarray;
triangleidx *d_triangles;

void cudaAllocateAndCopy(size_t pointsSize,
                         size_t triangleSize,

                         const triangleidx *triangles)
{
  // Allocate all the memory needed
  cudaMalloc(&d_pointarray, pointsSize);
  cudaMalloc(&d_triangles, triangleSize);

  // Copy the triangles index array, since its always static
  cudaMemcpy(d_triangles, triangles, triangleSize, cudaMemcpyHostToDevice);
}

void cudaCleanup()
{
  cudaFree(d_pointarray);
  cudaFree(d_triangles);
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
    size_t triangleNum,

    size_t pointarraySize,
    size_t trianglesSize,

    const int bgColor)
{
  ZoneScopedN("rayTrace function");

  // cudaMemcpy(d_pixelBuffer, pixelBuffer, pixelBufferSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pointarray, pointarray, pointarraySize, cudaMemcpyHostToDevice);

  constexpr dim3 blockDim(16, 16);
  constexpr dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

  rayTraceKernel<<<gridDim, blockDim>>>(
      pixelBuffer,
      camPos, camViewOrigin, imageX, imageY,
      inverseWidthMinus, inverseHeightMinus,
      d_pointarray, d_triangles, triangleNum,
      WIDTH, HEIGHT,
      bgColor);
}