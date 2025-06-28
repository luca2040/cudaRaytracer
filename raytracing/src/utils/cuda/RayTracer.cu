#include <iostream>
#include <cstdint>

#include "RayTracer.cuh"
#include "../../math/Definitions.h"
#include "../DrawValues.h"

__global__ void rayTraceKernel(
    uint32_t *pixelBuffer,
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

  // ray currentRay = ray(
  //     camPos,
  //     camViewOrigin + imageX * (static_cast<float>(x) * inverseWidthMinus) + imageY * (static_cast<float>(y) * inverseHeightMinus) - camPos);

  // // Bruteforce all the triangles
  // float currentZbuf = INFINITY;
  // for (size_t i = 0; i < triangleNum; i++)
  // {
  //   triangleidx triangle = triangles[i];

  //   float3_L v1 = pointarray[triangle.v1];
  //   float3_L v2 = pointarray[triangle.v2];
  //   float3_L v3 = pointarray[triangle.v3];
  //   unsigned int color = triangle.col;

  //   std::optional<float3_L> intersectionPoint = ray_intersects_triangle(currentRay.origin, currentRay.direction, v1, v2, v3);
  //   if (intersectionPoint)
  //   {
  //     float3_L intrstPoint = intersectionPoint.value();

  //     float distanceToCamera = distance(camPos, intrstPoint);
  //     if (distanceToCamera < currentZbuf)
  //     {
  //       currentZbuf = distanceToCamera;
  //       pixel_ptr[y * (texturePitch / 4) + x] = color;
  //     }
  //   }
  // }

  pixelBuffer[y * texturePitchFourth + x] = x * y / 100 % 0xFFFFFF; // Something just to test
}

void rayTrace(
    uint32_t *pixelBuffer,
    int texturePitch,

    const float3_L camPos,
    const float3_L camViewOrigin,
    const float3_L imageX,
    const float3_L imageY,
    float inverseWidthMinus,
    float inverseHeightMinus,

    const float3_L *pointarray,
    const triangleidx *triangles,
    size_t triangleNum,

    size_t pointarraySize,
    size_t trianglesSize)
{
  // ######## Device parameters ########

  uint32_t *d_pixelBuffer;
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

  // Get results back

  cudaDeviceSynchronize();
  cudaMemcpy(pixelBuffer, d_pixelBuffer, pixelBufferSize, cudaMemcpyDeviceToHost);

  // Clean up

  cudaFree(d_pixelBuffer);
  cudaFree(d_pointarray);
  cudaFree(d_triangles);
}