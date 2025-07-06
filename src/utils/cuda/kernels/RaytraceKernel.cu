#include "RaytraceKernel.cuh"

// Just while testing use these unoptimized functions
__device__ int colorMix(int color1, int color2, float t)
{
  int r1 = (color1 >> 16) & 0xFF;
  int g1 = (color1 >> 8) & 0xFF;
  int b1 = color1 & 0xFF;

  int r2 = (color2 >> 16) & 0xFF;
  int g2 = (color2 >> 8) & 0xFF;
  int b2 = color2 & 0xFF;

  int r = static_cast<int>((1 - t) * r1 + t * r2);
  int g = static_cast<int>((1 - t) * g1 + t * g2);
  int b = static_cast<int>((1 - t) * b1 + t * b2);

  return (r << 16) | (g << 8) | b;
}

__device__ float3_L reflectRay(float3_L rayDir, float3_L normal)
{
  if (dot3_cuda(rayDir, normal) > 0.0f)
    normal = normal * -1.0f;

  return rayDir - normal * (2.0f * dot3_cuda(rayDir, normal));
}

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

  float3_L rawDirection = camViewOrigin + imageX * (static_cast<float>(x) * inverseWidthMinus) + imageY * (static_cast<float>(y) * inverseHeightMinus) - camPos;

  ray currentRay = ray(
      camPos,
      normalize3_cuda(rawDirection));

  int currentColor = bgColor;
  float lastReflectiveness = 1.0f;

  for (size_t iteration = 0; iteration < RAY_HITS_MAX; iteration++)
  {
    // Bruteforce all the triangles
    float currentZbuf = INFINITY;
    triangleidx hitTriangle;
    float3_L lastHit;

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
        hitTriangle = triangle;
        lastHit = rayHit;
        // pixelBuffer[y * imageWidth + x] = make_uchar4_from_int(triangle.col);
      }
    }

    if (currentZbuf == INFINITY)
    {
      pixelBuffer[y * imageWidth + x] = make_uchar4_from_int(colorMix(currentColor, bgColor, lastReflectiveness));
      return;
    }

    currentColor = colorMix(currentColor, hitTriangle.col, lastReflectiveness);
    lastReflectiveness = hitTriangle.reflectiveness;
    if (lastReflectiveness < EPSILON)
      continue;

    currentRay.direction = reflectRay(currentRay.direction, hitTriangle.normal);
    currentRay.origin = lastHit;
  }

  pixelBuffer[y * imageWidth + x] = make_uchar4_from_int(currentColor);
}
