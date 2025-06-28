#pragma once

#include <cstdint>
#include "../../math/Definitions.h"

extern "C"
{
  void cudaAllocateAndCopy(size_t pointsSize,
                           size_t triangleSize,
                           size_t pixelBufferSize,

                           const triangleidx *triangles);

  void cudaCleanup();

  void rayTrace(
      uint32_t *pixelBuffer,
      int texturePitch,

      float3_L camPos,
      float3_L camViewOrigin,
      float3_L imageX,
      float3_L imageY,
      float inverseWidthMinus,
      float inverseHeightMinus,

      const float3_L *pointarray,
      size_t triangleNum,

      size_t pointarraySize,
      size_t trianglesSize,
      size_t pixelBufferSize,

      const unsigned int bgColor);
}
