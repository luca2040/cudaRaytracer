#pragma once

#include "../../DrawValues.h"
#include "../../../math/Definitions.h"
#include "../../../math/cuda/CudaMath.cuh"

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

    const uchar4 bgColor);