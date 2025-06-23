#ifndef DRAWING_UTILS
#define DRAWING_UTILS

#include <SDL2/SDL.h>
#include "../math/names.h"

void rasterizeFullTriangle(
    float3 v1,
    float3 v2,
    float3 v3,
    Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
    int color);

#endif