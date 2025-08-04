#pragma once

#include <cstdint>
#include "../../scene/structs/SceneObject.h"
#include "../../math/Definitions.h"

void cudaAllocateScene();
void cudaCleanup();
void rayTrace(uchar4 *pixelBuffer, int renderWidth, int renderHeight, uint frame);
