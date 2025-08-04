#include "RaytraceKernel.cuh"
#include "TracingKernel.cuh"
#include "../definitions/RenderingStructs.cuh"
#include "../../../math/cuda/CudaRNG.cuh"

__global__ void rayTraceKernel(
    uchar4 *pixelBuffer,

    Scene *scene,

    const int imageWidth,
    const int imageHeight,

    uint frame)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= imageWidth || y >= imageHeight)
    return;

  auto &cam = scene->cam;

  uint RNGState = (x + y * imageWidth) * frame;
  float3_L sampledColor = make_float3_L(0.0f, 0.0f, 0.0f);

  int samples = scene->simpleRender ? 1 : scene->samplesPerPixel;
  float sampleRange = scene->simpleRender ? 0.0f : scene->pixelSampleRange;

  for (int sample = 0; sample < samples; sample++)
  {
    float3_L xPercent = cam.imageX * ((static_cast<float>(x) + balancedRandomValue(RNGState) * sampleRange) * cam.inverseWidthMinus);
    float3_L yPercent = cam.imageY * ((static_cast<float>(y) + balancedRandomValue(RNGState) * sampleRange) * cam.inverseHeightMinus);

    float3_L rawDirection = cam.camViewOrigin + xPercent + yPercent - cam.camPos;

    Ray currentRay = Ray(cam.camPos, normalize3_cuda(rawDirection));
    RayData currentRayData;

    traceRay(scene, RNGState,
             currentRay, currentRayData);

    sampledColor = sampledColor + (currentRayData.color / static_cast<float>(samples));
  }

  uchar4 &currentPixelBuff = pixelBuffer[y * imageWidth + x];
  float3_L &currentAccumBuff = scene->d_accumulationBuffer[y * imageWidth + x];

  if (!scene->accumulate)
  {
    currentAccumBuff = sampledColor;
    currentPixelBuff = make_uchar4_from_f3l(sampledColor);
    return;
  }

  currentAccumBuff = currentAccumBuff + sampledColor;

  currentPixelBuff = make_uchar4_from_f3l(currentAccumBuff / scene->accumulatedFrames);
}
