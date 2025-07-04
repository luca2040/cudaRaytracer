#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <iostream>

#include "../math/Definitions.h"
#include "../math/Operations.h"
#include "../scene/composition/SceneCompositor.h"
#include "DrawLoop.h"

#include "opengl/Shader.h"
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

void onSceneComposition()
{
  composeScene(points, pointsCount,
               triangles, triangleNum,
               trIndexPairs, trIndexPairCount,
               drawLoopValues);
}

cudaGraphicsResource *cudaPboResource;
GLuint shaderProgram;
GLuint quadVAO = 0, quadVBO = 0;

void onSetupFrame(GLuint pbo)
{
  pointsSize = sizeof(float3_L) * pointsCount;
  triangleSize = sizeof(triangleidx) * triangleNum;

  cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
  setupQuad(quadVAO, quadVBO);
  shaderProgram = createShaderProgram();

  cudaAllocateAndCopy(pointsSize, triangleSize, triangles);
}

void onClose()
{
  cudaCleanup();

  glDeleteProgram(shaderProgram);
  glDeleteVertexArrays(1, &quadVAO);
  glDeleteBuffers(1, &quadVBO);
}

void drawFrame(GLuint tex, GLuint pbo)
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

  // Map the PBO for CUDA access

  TracyCZoneN(PboMap, "PBO map", true);

  cudaGraphicsMapResources(1, &cudaPboResource, 0);

  uchar4 *pxlsPtr;
  size_t pxlsPtrSize;
  cudaGraphicsResourceGetMappedPointer((void **)&pxlsPtr, &pxlsPtrSize, cudaPboResource);

  TracyCZoneEnd(PboMap);

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

  rayTrace(pxlsPtr,
           camPos, camViewOrigin,
           imageX, imageY,
           inverseWidthMinus, inverseHeightMinus,
           pointarray, triangleNum,
           pointsSize, triangleSize,
           BG_COLOR);

  TracyCZoneEnd(raytraceFunction);

  // Clean up

  delete[] pointarray;

  // Unlock and render texture

  TracyCZoneN(finalCleanup, "Texture render to screen", true);

  cudaGraphicsUnmapResources(1, &cudaPboResource);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(shaderProgram);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex);
  glUniform1i(glGetUniformLocation(shaderProgram, "tex"), 0);

  glBindVertexArray(quadVAO);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);

  TracyCZoneEnd(finalCleanup);

  FrameMark;
}

void keyPressed(SDL_Keycode key)
{
}