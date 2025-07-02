#include <SDL2/SDL.h>
#include <iostream>

#include "../math/Definitions.h"
#include "../math/Operations.h"
#include "../scene/composition/SceneCompositor.h"
#include "DrawLoop.h"

#include "cuda/RayTracer.cuh"

#include "../third_party/tracy/tracy/Tracy.hpp"
#include "../third_party/tracy/tracy/TracyC.h"

// ######################### Camera settings ##########################

const float camFOVdeg = 60;

float camZoom = 2.0f;
float3_L camPos = {0, 0, 0};
float3_L camLookingPoint = {0, 0, 1};

// ####################################################################

const float camFOV = camFOVdeg * M_PI / 180.0f;
const float imagePlaneHeight = 2.0f * tan(camFOV / 2.0f);
const float imagePlaneWidth = imagePlaneHeight * ASPECT;

// ########### Variables initialized on start ###########

DrawingLoopValues drawLoopValues;

float3_L *points;
triangleidx *triangles;
transformIndexPair *trIndexPairs;

size_t pointsCount;
size_t triangleNum;
size_t trIndexPairCount;

size_t pointsSize;
size_t triangleSize;
size_t pixelBufferSize;

void onSceneComposition()
{
  composeScene(points, pointsCount,
               triangles, triangleNum,
               trIndexPairs, trIndexPairCount,
               drawLoopValues);
}

void onSetupFrame(SDL_Renderer *renderer, SDL_Texture *texture)
{
  pointsSize = sizeof(float3_L) * pointsCount;
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
  ZoneScopedN("drawFrame function");

  // Rotations

  TracyCZoneN(timeFromSDL, "Get movements from SDL time", true);

  Uint32 time = SDL_GetTicks();
  float xrot = fmod((static_cast<float>(time) * 0.0005f), TWO_PI);
  float yrot = fmod((static_cast<float>(time) * 0.001f), TWO_PI);

  float ymov = std::sin(yrot) * 1.25f;

  TracyCZoneEnd(timeFromSDL);

  // Copy vertexes array

  TracyCZoneN(copyPointArray, "Copy points array", true);

  float3_L *pointarray = new float3_L[pointsCount];
  std::copy(points, points + pointsCount, pointarray);

  TracyCZoneEnd(copyPointArray);

  // Custom single-object rotations apply

  TracyCZoneN(setRotationsToObjects, "Set rotations to objects", true);

  trIndexPairs[drawLoopValues.simpleCubeIndex].transform.rotationAngles = {xrot, yrot, 0.0f};
  trIndexPairs[drawLoopValues.movingCubeIndex].transform.relativePos = {0.0f, ymov, 0.0f};

  TracyCZoneEnd(setRotationsToObjects);

  // Apply rotations to copied list

  TracyCZoneN(matRotateVerts, "Matrix rotate vertices", true);

  for (size_t indexPairI = 0; indexPairI < trIndexPairCount; indexPairI++)
  {
    transformIndexPair currentPair = trIndexPairs[indexPairI];
    mat3x3 currentMat = currentPair.transform.getRotationMatrix();

    for (size_t i = currentPair.startIdx; i < currentPair.endIdx; i++)
    {
      pointarray[i] -= currentPair.transform.rotationCenter;
      pointarray[i] = currentMat * pointarray[i];
      pointarray[i] += currentPair.transform.rotationCenter + currentPair.transform.relativePos;
    }
  }

  TracyCZoneEnd(matRotateVerts);

  // Lock texture

  TracyCZoneN(lockTexture, "Lock texture", true);

  void *pixels;
  int texturePitch;
  SDL_LockTexture(texture, NULL, &pixels, &texturePitch);

  Uint32 *pixel_ptr = (Uint32 *)pixels;
  memset(pixel_ptr, 0, texturePitch * HEIGHT);

  TracyCZoneEnd(lockTexture);

  // Camera setup

  TracyCZoneN(cameraSettings, "Camera settings", true);

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

  TracyCZoneEnd(cameraSettings);

  // Generate and trace rays

  TracyCZoneN(raytraceFunction, "drawFrame rayTrace function call", true);

  rayTrace(pixel_ptr, texturePitch,
           camPos, camViewOrigin,
           imageX, imageY,
           inverseWidthMinus, inverseHeightMinus,
           pointarray, triangleNum,
           pointsSize, triangleSize, pixelBufferSize,
           BG_COLOR);

  TracyCZoneEnd(raytraceFunction);

  // Clean up

  delete[] pointarray;

  // Unlock and render texture

  TracyCZoneN(finalCleanup, "SDL texture render", true);

  SDL_UnlockTexture(texture);
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, NULL, NULL);

  TracyCZoneEnd(finalCleanup);

  FrameMark;
}

void keyPressed(SDL_Keycode key)
{
}