#include "RaytraceKernel.cuh"
#include "TracingKernel.cuh"
#include "../definitions/RenderingStructs.cuh"
#include "../../../math/cuda/CudaRNG.cuh"

__global__ void rayTraceKernel(
    uchar4 *pixelBuffer,

    Scene *scene,

    const int imageWidth,
    const int imageHeight)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= imageWidth || y >= imageHeight)
    return;

  auto &cam = scene->cam;

  uint RNGState = x + y * imageWidth;
  float3_L accumulatedColor = make_float3_L(0.0f, 0.0f, 0.0f);

  for (int sample = 0; sample < scene->samplesPerPixel; sample++)
  {
    float3_L xPercent = cam.imageX * ((static_cast<float>(x) + (randomValue(RNGState) * 2.0f - 1.0f) * scene->pixelSampleRange) * cam.inverseWidthMinus);
    float3_L yPercent = cam.imageY * ((static_cast<float>(y) + (randomValue(RNGState) * 2.0f - 1.0f) * scene->pixelSampleRange) * cam.inverseHeightMinus);

    float3_L rawDirection = cam.camViewOrigin + xPercent + yPercent - cam.camPos;

    Ray currentRay = Ray(cam.camPos, normalize3_cuda(rawDirection));
    RayData currentRayData;

    traceRay(scene,
             currentRay, currentRayData);

    accumulatedColor = accumulatedColor + (currentRayData.color / static_cast<float>(scene->samplesPerPixel));
  }

  pixelBuffer[y * imageWidth + x] = make_uchar4_from_f3l(accumulatedColor);
}
