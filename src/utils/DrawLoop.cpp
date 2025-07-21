#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
// #include <omp.h>

#include <iostream>

#include "../math/Definitions.h"
#include "../math/Operations.h"
#include "../scene/composition/SceneCompositor.h"
#include "DrawLoop.h"
#include "KeyBinds.h"

#include "gui/GuiWindow.h"
#include "../scene/structs/Scene.h"

#include "opengl/Shader.h"
#include "cuda/RayTracer.cuh"

#include "../third_party/tracy/tracy/Tracy.hpp"
#include "../third_party/tracy/tracy/TracyC.h"

void onSceneComposition()
{
  composeScene(scene.points, scene.pointsCount,
               scene.triangles, scene.triangleNum,
               scene.trIndexPairs, scene.trIndexPairCount,
               scene.dyntriangles, scene.dyntrianglesNum);
}

cudaGraphicsResource *cudaPboResource;
GLuint shaderProgram;
GLuint quadVAO = 0, quadVBO = 0;

void onSetupFrame(GLuint pbo)
{
  scene.pointsSize = sizeof(float3_L) * scene.pointsCount;
  scene.triangleSize = sizeof(triangleidx) * scene.triangleNum;

  cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
  setupQuad(quadVAO, quadVBO);
  shaderProgram = createShaderProgram();

  cudaAllocateAndCopy(scene.pointsSize, scene.triangleSize);
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

  // Copy vertexes array

  float3_L *pointarray = new float3_L[scene.pointsCount];
  std::copy(scene.points, scene.points + scene.pointsCount, pointarray);

  // Time for transforms

  Uint32 time = SDL_GetTicks();

  // Apply rotations to copied list

  TracyCZoneN(matRotateVerts, "Matrix rotate vertices", true);

  // Rotate vertices and apply transforms
  // #pragma omp parallel for
  for (size_t indexPairI = 0; indexPairI < scene.trIndexPairCount; indexPairI++)
  {
    transformIndexPair currentPair = scene.trIndexPairs[indexPairI];
    ObjTransform currentTransform = currentPair.transform;

    if (currentTransform.hasTransformFunction)
    {
      currentTransform.trFunc(time, currentTransform.rotationAngles, currentTransform.relativePos);
    }

    mat3x3 currentMat = currentTransform.getRotationMatrix();

    float3_L rotCenter = currentTransform.rotationCenter;
    float3_L centerShifted = rotCenter + currentTransform.relativePos;

    for (size_t i = currentPair.startIdx; i < currentPair.endIdx; i++)
    {
      float3_L &pt = pointarray[i];

      pt -= rotCenter;
      pt = currentMat * pt;
      pt += centerShifted;
    }
  }

  // Recalculate normals
  for (size_t i = 0; i < scene.dyntrianglesNum; i++)
  {
    triangleidx &currentTriangle = scene.triangles[scene.dyntriangles[i]];

    float3_L v1 = pointarray[currentTriangle.v1];
    float3_L v2 = pointarray[currentTriangle.v2];
    float3_L v3 = pointarray[currentTriangle.v3];

    float3_L side1 = v2 - v1;
    float3_L side2 = v3 - v1;

    currentTriangle.normal = normalize(cross3(side1, side2));
  }

  TracyCZoneEnd(matRotateVerts);

  // Map the PBO for CUDA access

  cudaGraphicsMapResources(1, &cudaPboResource, 0);

  uchar4 *pxlsPtr;
  size_t pxlsPtrSize;
  cudaGraphicsResourceGetMappedPointer((void **)&pxlsPtr, &pxlsPtrSize, cudaPboResource);

  // Camera setup

  TracyCZoneN(cameraSettings, "Camera settings", true);

  auto &cam = scene.cam;

  float cx = cos(cam.camXrot), sx = sin(cam.camXrot);
  float cy = cos(cam.camYrot), sy = sin(cam.camYrot);

  float3_L cameraDirVec = {0, sy, cy};

  mat3x3 yRotMat = {
      float3_L(cx, 0.0f, sx),
      float3_L(0.0f, 1.0f, 0.0f),
      float3_L(-sx, 0.0f, cx)};

  cam.camForward = yRotMat * cameraDirVec;
  cam.camRight = normalize(cross3(cam.camForward, {0, -1, 0})); // camForward, worldUp
  cam.camUp = cross3(cam.camRight, cam.camForward);

  // Camera settings declaration

  float camFOV = cam.camFOVdeg * M_PI / 180.0f;
  float imagePlaneHeight = 2.0f * tan(camFOV / 2.0f);
  float imagePlaneWidth = imagePlaneHeight * ASPECT;

  // Camera placement

  float3_L imageCenter = cam.camPos + cam.camForward;
  float3_L imageX = cam.camRight * imagePlaneWidth * cam.camZoom;
  float3_L imageY = cam.camUp * imagePlaneHeight * cam.camZoom;
  float3_L camViewOrigin = imageCenter - imageX * 0.5f - imageY * 0.5f;

  constexpr float inverseWidthMinus = 1.0f / static_cast<float>(WIDTH - 1);
  constexpr float inverseHeightMinus = 1.0f / static_cast<float>(HEIGHT - 1);

  TracyCZoneEnd(cameraSettings);

  // Generate and trace rays

  rayTrace(pxlsPtr,
           cam.camPos, camViewOrigin,
           imageX, imageY,
           inverseWidthMinus, inverseHeightMinus,
           pointarray, scene.triangles, scene.triangleNum,
           scene.pointsSize, scene.triangleSize,
           BG_COLOR);

  // Clean up

  delete[] pointarray;

  // Unlock and render texture

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

  FrameMark;
}

void keyPressed(SDL_Keycode key)
{
}

void mouseMoved(int2_L &mouse, int2_L &pMouse)
{
  Uint32 mouseState = SDL_GetMouseState(nullptr, nullptr);
  int2_L dMouse = mouse - pMouse;

  if (mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT))
  {
    auto &cam = scene.cam;

    cam.camXrot = fmod(cam.camXrot += 0.00075f * dMouse.x, TWO_PI);
    cam.camYrot = clamp(cam.camYrot += 0.001f * dMouse.y, cameraVerticalMinRot, cameraVerticalMaxRot);
  }
}

void checkForKeys()
{
  const Uint8 *keyState = SDL_GetKeyboardState(NULL);

  float sameDirMov = (keyState[forwardKey] ? 1.0f : 0.0f) - (keyState[backKey] ? 1.0f : 0.0f);
  float rightDirMov = (keyState[rightKey] ? 1.0f : 0.0f) - (keyState[leftKey] ? 1.0f : 0.0f);

  if (sameDirMov == 0.0f && rightDirMov == 0.0f)
    return;

  auto &cam = scene.cam;

  float3_L frontMovement = cam.camForward * sameDirMov;
  float3_L rightMovement = cam.camRight * rightDirMov;

  float3_L totalCamMov = normalize(frontMovement + rightMovement) * (movingSpeed * guiWindow.io->DeltaTime);
  cam.camPos += totalCamMov;
}