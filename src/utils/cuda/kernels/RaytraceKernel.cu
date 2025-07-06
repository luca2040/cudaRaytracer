#include "RaytraceKernel.cuh"

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

    int bgColor)
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

    reflectRay(currentRay.direction, hitTriangle.normal);
    currentRay.origin = lastHit;
  }

  pixelBuffer[y * imageWidth + x] = make_uchar4_from_int(currentColor);
}
