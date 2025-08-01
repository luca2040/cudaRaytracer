#pragma once

#include "../../DrawValues.h"
#include "../../../math/Definitions.h"
#include "../../../scene/structs/Scene.h"
#include "../../../math/cuda/CudaMath.cuh"
#include "../definitions/RenderingStructs.cuh"

__global__ void rayTraceKernel(
    uchar4 *pixelBuffer,

    Scene *scene,

    const int imageWidth,
    const int imageHeight);