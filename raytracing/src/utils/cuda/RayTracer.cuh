#pragma once

#include <SDL2/SDL.h>
#include "../../math/Definitions.h"

void rayTrace(
    Uint32 *pixelBuffer,
    int texturePitch,

    float3_L camPos,
    float3_L camViewOrigin,
    float3_L imageX,
    float3_L imageY,
    float inverseWidthMinus,
    float inverseHeightMinus,

    const float3_L *pointarray,
    const triangleidx *triangles,
    size_t triangleNum,

    size_t pointarraySize,
    size_t trianglesSize);