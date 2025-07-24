#include "RaytraceKernel.cuh"
#include "TracingKernel.cuh"
#include "../definitions/RenderingStructs.cuh"

__global__ void rayTraceKernel(
    uchar4 *pixelBuffer,

    Scene *scene,

    const int imageWidth,
    const int imageHeight,

    const float3_L bgColor)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= imageWidth || y >= imageHeight)
    return;

  auto &cam = scene->cam;

  float3_L rawDirection = cam.camViewOrigin + cam.imageX * (static_cast<float>(x) * cam.inverseWidthMinus) + cam.imageY * (static_cast<float>(y) * cam.inverseHeightMinus) - cam.camPos;

  ray currentRay = ray(
      cam.camPos,
      normalize3_cuda(rawDirection));
  RayData currentRayData;

  traceRay(scene,
           currentRay, currentRayData,
           bgColor);

  pixelBuffer[y * imageWidth + x] = make_uchar4_from_f3l(currentRayData.color);
}
