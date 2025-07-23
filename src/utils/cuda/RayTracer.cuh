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
    const int bgColor);
