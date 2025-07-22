#pragma once

#include <cstdint>
#include "../../scene/structs/SceneObject.h"
#include "../../math/Definitions.h"

void cudaAllocateAndCopy(size_t pointsSize,
                         size_t triangleSize,
                         size_t sceneobjectsSize);

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
    const triangleidx *triangles,
    const SceneObject *sceneobjects,
    size_t sceneobjectsNum,

    size_t pointarraySize,
    size_t trianglesSize,
    size_t sceneobjectsSize,

    const int bgColor);
