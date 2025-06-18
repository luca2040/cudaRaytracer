#include <SDL2/SDL.h>

void depthFillTriangle(
    float x1, float y1, float z1,
    float x2, float y2, float z2,
    float x3, float y3, float z3,
    Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
    int color,
    int WIDTH, int HEIGHT);