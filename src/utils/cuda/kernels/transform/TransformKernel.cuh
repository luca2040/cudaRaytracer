#pragma once

#include "../../../../scene/structs/Scene.h"

__global__ void resetAABBsKernel(Scene *scene);
__global__ void transformKernel(Scene *scene);
__global__ void normalComputeKernel(Scene *scene);