#include <iostream>

#include "RayTracer.cuh"
#include "../../math/Definitions.h"
#include "../DrawLoop.h"

__global__ void rayTraceKernel(
    Uint32 *pixelBuffer,
    int texturePitchFourth,

    float3_L camPos,
    float3_L camViewOrigin,
    float3_L imageX,
    float3_L imageY,
    float inverseWidthMinus,
    float inverseHeightMinus,

    const float3_L *pointarray,
    const triangleidx *triangles,
    size_t triangleNum,

    int imageWidth,
    int imageHeight)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= imageWidth || y >= imageHeight)
    return;
}

void rayTrace(
    Uint32 *pixelBuffer,
    int texturePitch,

    const float3_L &camPos,
    const float3_L &camViewOrigin,
    const float3_L &imageX,
    const float3_L &imageY,
    float inverseWidthMinus,
    float inverseHeightMinus,

    const float3_L *pointarray,
    const triangleidx *triangles,
    size_t triangleNum,

    size_t pointarraySize,
    size_t trianglesSize)
{
  // ######## Device parameters ########
  Uint32 *d_pixelBuffer;
  float3_L *d_pointarray;
  triangleidx *d_triangles;

  size_t pixelBufferSize = texturePitch * HEIGHT;

  cudaMalloc(&d_pixelBuffer, pixelBufferSize);
  cudaMalloc(&d_pointarray, pointarraySize);
  cudaMalloc(&d_triangles, trianglesSize);

  cudaMemcpy(d_pixelBuffer, pixelBuffer, pixelBufferSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pointarray, pointarray, pointarraySize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_triangles, triangles, trianglesSize, cudaMemcpyHostToDevice);

  dim3 blockDim(16, 16);
  dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

  rayTraceKernel<<<gridDim, blockDim>>>(
      d_pixelBuffer,
      texturePitch / 4,
      camPos, camViewOrigin, imageX, imageY,
      inverseWidthMinus, inverseHeightMinus,
      d_pointarray, d_triangles, triangleNum,
      WIDTH, HEIGHT);
}