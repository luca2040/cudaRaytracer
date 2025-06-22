#include <SDL2/SDL.h>
#include <glm/glm.hpp>

void rasterizeFullTriangle(
    glm::vec3 v1,
    glm::vec3 v2,
    glm::vec3 v3,
    Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
    int color);