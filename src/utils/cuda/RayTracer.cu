#include <iostream>
#include <cstdint>

#include "RayTracer.cuh"
#include "../DrawValues.h"
#include "../../math/Definitions.h"
#include "../../math/cuda/CudaMath.cuh"

#include "../../third_party/tracy/tracy/Tracy.hpp"

__global__ void rayTraceKernel(
    uchar4 *pixelBuffer,

    float3_L camPos,
    float3_L camViewOrigin,
    float3_L imageX,
    float3_L imageY,
    float inverseWidthMinus,
    float inverseHeightMinus,

    const float3_L *pointarray,
    const triangleidx *triangles,
    size_t triangleNum,

    const int imageWidth,
    const int imageHeight,

    const int bgColor)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= imageWidth || y >= imageHeight)
    return;

  ray currentRay = ray(
      camPos,
      camViewOrigin + imageX * (static_cast<float>(x) * inverseWidthMinus) + imageY * (static_cast<float>(y) * inverseHeightMinus) - camPos);

  // Bruteforce all the triangles
  float currentZbuf = INFINITY;
  for (size_t i = 0; i < triangleNum; i++)
  {
    triangleidx triangle = triangles[i];

    float t, u, v;
    float3_L rayHit;

    bool hasIntersected = rayTriangleIntersection(currentRay,
                                                  pointarray[triangle.v1], pointarray[triangle.v2], pointarray[triangle.v3],
                                                  t, u, v,
                                                  rayHit);

    if (hasIntersected && (t < currentZbuf))
    {
      currentZbuf = t;
      pixelBuffer[y * imageWidth + x] = make_uchar4_from_int(triangle.col);
    }
  }

  if (currentZbuf == INFINITY)
  {
    pixelBuffer[y * imageWidth + x] = make_uchar4_from_int(bgColor);
  }
}

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