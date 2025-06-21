#include <SDL2/SDL.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../math/mathUtils.h"
#include "DrawLoop.h"
#include "DrawingUtils.h"

// Camera settings
float camZ = 600;

// Rotations
glm::vec3 rotcenter(0.0f, 0.0f, 3000.0f);

glm::vec3 points[] = {
    // Front face
    {-1000.0f, -1000.0f, 2000.0f},
    {1000.0f, -1000.0f, 2000.0f},
    {1000.0f, 1000.0f, 2000.0f},
    {-1000.0f, 1000.0f, 2000.0f},
    // Back face
    {-1000.0f, -1000.0f, 4000.0f},
    {1000.0f, -1000.0f, 4000.0f},
    {1000.0f, 1000.0f, 4000.0f},
    {-1000.0f, 1000.0f, 4000.0f}};

// Vertex intexes [3], color
glm::uvec4 triangles[] = {
    {0, 1, 3, 0xFF0000},
    {1, 3, 2, 0xFF0000},

    {1, 5, 2, 0x00FF00},
    {5, 2, 6, 0x00FF00},

    {0, 3, 4, 0x0000FF},
    {4, 3, 7, 0x0000FF},

    {4, 7, 5, 0xFFFF00},
    {5, 7, 6, 0xFFFF00},

    {0, 1, 4, 0xFF00FF},
    {1, 4, 5, 0xFF00FF},

    {3, 2, 7, 0x00FFFF},
    {2, 7, 6, 0x00FFFF},
};

void drawFrame(SDL_Renderer *renderer, SDL_Texture *texture)
{
  // Rotations

  constexpr float TWO_PI = 2.0f * M_PI;

  Uint32 time = SDL_GetTicks();
  float xrot = fmod((static_cast<float>(time) * 0.0005f), TWO_PI);
  float yrot = fmod((static_cast<float>(time) * 0.001f), TWO_PI);

  // Copy vertexes array

  constexpr size_t pointsCount = sizeof(points) / sizeof(points[0]);

  glm::vec3 pointarray[pointsCount];
  std::copy(std::begin(points), std::end(points), pointarray);

  glm::mat3 xrotmat = glm::mat3(glm::rotate(glm::mat4(1.0f), xrot, glm::vec3(1, 0, 0)));
  glm::mat3 yrotmat = glm::mat3(glm::rotate(glm::mat4(1.0f), yrot, glm::vec3(0, 1, 0)));
  glm::mat3 rotCombined = xrotmat * yrotmat;

  for (size_t i = 0; i < pointsCount; i++)
  {

    // Vertex calculations and projection all compressed into a single cycle now

    pointarray[i] -= rotcenter;
    pointarray[i] = rotCombined * pointarray[i];
    pointarray[i] += rotcenter;

    float depthCamInverse = 1.0f / (camZ + pointarray[i].z);

    pointarray[i].x = pointarray[i].x * depthCamInverse * camZ + HALF_WIDTH;
    pointarray[i].y = pointarray[i].y * depthCamInverse * camZ + HALF_HEIGHT;
  }

  // Lock texture

  void *pixels;
  int texturePitch;
  SDL_LockTexture(texture, NULL, &pixels, &texturePitch);

  Uint32 *pixel_ptr = (Uint32 *)pixels;
  memset(pixel_ptr, 0, texturePitch * HEIGHT);

  // Init depth buffer

  float drawDepthBuffer[TOTAL_PIXELS];
  std::fill_n(drawDepthBuffer, TOTAL_PIXELS, INFINITY);

  // To use this: drawDepthBuffer[y * WIDTH + x]

  // Rasterize the triangles

  constexpr size_t triangleNum = sizeof(triangles) / sizeof(triangles[0]);

  for (size_t i = 0; i < triangleNum; i++)
  {
    // The x and y are the projected ones, the z is for the depth buffer

    float x1 = pointarray[triangles[i][0]][0];
    float y1 = pointarray[triangles[i][0]][1];
    float z1 = pointarray[triangles[i][0]][2];

    float x2 = pointarray[triangles[i][1]][0];
    float y2 = pointarray[triangles[i][1]][1];
    float z2 = pointarray[triangles[i][1]][2];

    float x3 = pointarray[triangles[i][2]][0];
    float y3 = pointarray[triangles[i][2]][1];
    float z3 = pointarray[triangles[i][2]][2];

    int color = triangles[i][3];

    depthFillTriangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, pixel_ptr, texturePitch, drawDepthBuffer, color, WIDTH, HEIGHT);
  }

  // Unlock and render texture

  SDL_UnlockTexture(texture);
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, NULL, NULL);
}

void keyPressed(SDL_Keycode key)
{
}