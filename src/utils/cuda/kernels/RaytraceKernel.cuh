#pragma once

#include "../../DrawValues.h"
#include "../../../math/Definitions.h"
#include "../../../math/cuda/CudaMath.cuh"
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

    const float3_L bgColor);