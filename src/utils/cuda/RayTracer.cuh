#pragma once

#include <cstdint>
#include "../../math/Definitions.h"

void cudaAllocateAndCopy(size_t pointsSize,
                         size_t triangleSize,

                         const triangleidx *triangles);

void cudaCleanup();

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

    const int bgColor);
