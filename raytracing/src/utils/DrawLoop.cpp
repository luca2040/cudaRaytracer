#include <SDL2/SDL.h>
#include <iostream>

#include "../math/Definitions.h"
#include "../math/Operations.h"
#include "DrawLoop.h"

#include "cuda/RayTracer.cuh"

// ######################### Camera settings ##########################

const float camFOVdeg = 60;

float camZoom = 2.0f;
float3_L camPos = {0, 0, 0};
float3_L camLookingPoint = {0, 0, 1};

// ####################################################################

const float camFOV = camFOVdeg * M_PI / 180.0f;
const float imagePlaneHeight = 2.0f * tan(camFOV / 2.0f);
const float imagePlaneWidth = imagePlaneHeight * ASPECT;

// ####################################################################

// Objects
float3_L rotcenter(0.0f, 0.0f, 3000.0f);

float3_L points[] = {
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
triangleidx triangles[] = {
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

// ########### Variables to setup on start ###########

size_t pointsCount;
size_t pointsSize;

size_t triangleNum;
size_t triangleSize;

size_t pixelBufferSize;

void onSetupFrame(SDL_Renderer *renderer, SDL_Texture *texture)
{
  pointsCount = sizeof(points) / sizeof(points[0]);
  pointsSize = sizeof(float3_L) * pointsCount;

  triangleNum = sizeof(triangles) / sizeof(triangles[0]);
  triangleSize = sizeof(triangleidx) * triangleNum;

  // Get the texturePitch needed to allocate the right size on CUDA

  void *pixelsPlaceholder;
  int texturePitch;
  SDL_LockTexture(texture, NULL, &pixelsPlaceholder, &texturePitch);
  SDL_UnlockTexture(texture);

  pixelBufferSize = HEIGHT * texturePitch;

  cudaAllocateAndCopy(pointsSize, triangleSize, pixelBufferSize, triangles);
}

void onClose(SDL_Renderer *renderer, SDL_Texture *texture)
{
  cudaCleanup();
}

void drawFrame(SDL_Renderer *renderer, SDL_Texture *texture)
{
  // Rotations

  Uint32 time = SDL_GetTicks();
  float xrot = fmod((static_cast<float>(time) * 0.0005f), TWO_PI);
  float yrot = fmod((static_cast<float>(time) * 0.001f), TWO_PI);

  // Copy vertexes array

  float3_L *pointarray = new float3_L[pointsCount];
  std::copy(std::begin(points), std::end(points), pointarray);

  // Apply rotations to copied list

  mat3x3 yrotmat = {
      float3_L(cos(yrot), 0.0f, sin(yrot)),
      float3_L(0.0f, 1.0f, 0.0f),
      float3_L(-sin(yrot), 0.0f, cos(yrot))};

  mat3x3 xrotmat = {
      float3_L(1.0f, 0.0f, 0.0f),
      float3_L(0.0f, cos(xrot), -sin(xrot)),
      float3_L(0.0f, sin(xrot), cos(xrot))};

  mat3x3 rotCombined = xrotmat * yrotmat;

  for (size_t i = 0; i < pointsCount; i++)
  {
    pointarray[i] -= rotcenter;
    pointarray[i] = rotCombined * pointarray[i];
    pointarray[i] += rotcenter;
  }

  // Lock texture

  void *pixels;
  int texturePitch;
  SDL_LockTexture(texture, NULL, &pixels, &texturePitch);

  Uint32 *pixel_ptr = (Uint32 *)pixels;
  memset(pixel_ptr, 0, texturePitch * HEIGHT);

  // Camera setup

  float3_L camForward = normalize(camLookingPoint - camPos);
  float3_L camRight = normalize(cross3(camForward, {0, -1, 0})); // camForward, worldUp
  float3_L camUp = cross3(camRight, camForward);

  // Camera placement

  float3_L imageCenter = camPos + camForward;
  float3_L imageX = camRight * imagePlaneWidth * camZoom;
  float3_L imageY = camUp * imagePlaneHeight * camZoom;
  float3_L camViewOrigin = imageCenter - imageX * 0.5f - imageY * 0.5f;

  constexpr float inverseWidthMinus = 1.0f / static_cast<float>(WIDTH - 1);
  constexpr float inverseHeightMinus = 1.0f / static_cast<float>(HEIGHT - 1);

  // Generate and trace rays

  rayTrace(pixel_ptr, texturePitch,
           camPos, camViewOrigin,
           imageX, imageY,
           inverseWidthMinus, inverseHeightMinus,
           pointarray, triangleNum,
           pointsSize, triangleSize, pixelBufferSize,
           BG_COLOR);

  // Clean up

  delete[] pointarray;

  // Unlock and render texture

  SDL_UnlockTexture(texture);
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, NULL, NULL);
}

void keyPressed(SDL_Keycode key)
{
}