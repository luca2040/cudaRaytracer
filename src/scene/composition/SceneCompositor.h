#pragma once

#include "../../utils/DrawValues.h"
#include "../SceneBuilder.h"

void composeScene(float3_L *&pointarray, size_t &pointCount,
                  triangleidx *&triangles, size_t &triangleCount,
                  transformIndexPair *&indexpairs, size_t &indexPairCount);