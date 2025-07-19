#include "RaytraceKernel.cuh"
#include "TracingKernel.cuh"
#include "../definitions/RenderingStructs.cuh"

__global__ void rayTraceKernel(
    uchar4 *pixelBuffer,

    float3_L camPos,
    float3_L camViewOrigin,
    float3_L imageX,
    float3_L imageY,
    float inverseWidthMinus,
    float inverseHeightMinus,

    SceneMemoryPointers memPointers,

    const int imageWidth,
    const int imageHeight,

    const float3_L bgColor)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= imageWidth || y >= imageHeight)
    return;

  float3_L rawDirection = camViewOrigin + imageX * (static_cast<float>(x) * inverseWidthMinus) + imageY * (static_cast<float>(y) * inverseHeightMinus) - camPos;

  ray currentRay = ray(
      camPos,
      normalize3_cuda(rawDirection));
  RayData currentRayData;

  traceRay(memPointers,
           currentRay, currentRayData,
           bgColor);

  pixelBuffer[y * imageWidth + x] = make_uchar4_from_f3l(currentRayData.color);
}
