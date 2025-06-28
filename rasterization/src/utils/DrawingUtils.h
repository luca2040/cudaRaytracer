#ifndef DRAWING_UTILS
#define DRAWING_UTILS

#include <SDL2/SDL.h>
#include "../math/names.h"

void rasterizeFullTriangle(
    float3_L v1,
    float3_L v2,
    float3_L v3,
    Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
    int color);

#endif