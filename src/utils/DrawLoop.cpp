#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>

#include "../math/Definitions.h"
#include "../math/Operations.h"
#include "../scene/composition/SceneCompositor.h"
#include "DrawLoop.h"
#include "KeyBinds.h"

#include "gui/GuiWindow.h"
#include "../scene/structs/CameraCalcs.h"
#include "../scene/structs/Scene.h"

#include "opengl/Shader.h"
#include "cuda/RayTracer.cuh"
#include "cuda/ComputeTransforms.cuh"

#include "Profiling.h"

void onSceneComposition()
{
  composeScene();
}

cudaGraphicsResource *cudaPboResource;
GLuint shaderProgram;
GLuint quadVAO = 0, quadVBO = 0;

void onSetupFrame(GLuint pbo)
{
  scene->pointsSize = getItemSize<float3_L>(scene->pointsCount);
  scene->pointTableSize = getItemSize<size_t>(scene->pointsCount);
  scene->triangleSize = getItemSize<triangleidx>(scene->triangleNum);
  scene->sceneObjectsSize = getItemSize<SceneObject>(scene->sceneobjectsNum);
  scene->matricesSize = getItemSize<mat4x4>(scene->sceneobjectsNum);

  mat4x4 *transformMatrices = nullptr;
  mat4x4 *d_transformMatrices = nullptr;
  cudaHostAlloc((void **)&transformMatrices, scene->matricesSize, cudaHostAllocMapped);
  scene->transformMatrices = transformMatrices;
  cudaHostGetDevicePointer((void **)&d_transformMatrices, (void *)transformMatrices, 0);
  scene->d_transformMatrices = d_transformMatrices;

  cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
  setupQuad(quadVAO, quadVBO);
  shaderProgram = createShaderProgram();

  cudaAllocateScene();
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
  ZONESCOPEDNC("drawFrame function", PROFILER_BROWN);
  TRACYCZONENC(matRotateVerts, "Compute transform matrices", true, PROFILER_BLUE);

  Uint32 time = SDL_GetTicks();

  // Rotate vertices and apply transforms
  for (size_t indexPairI = 0; indexPairI < scene->trIndexPairCount; indexPairI++)
  {
    transformIndexPair currentPair = scene->trIndexPairs[indexPairI];
    ObjTransform currentTransform = currentPair.transform;

    mat4x4 &currentMatrix = scene->transformMatrices[currentPair.sceneObjectReference];

    if (currentTransform.hasTransformFunction)
    {
      currentTransform.trFunc(time, currentTransform.rotationAngles, currentTransform.relativePos);
    }

    float3_L rotCenter = currentTransform.rotationCenter;
    float3_L totalTranslation = rotCenter + currentTransform.relativePos;

    mat4x4 moveToCenterMat = translation4x4mat(0.0f - rotCenter);
    mat4x4 rotateMat = currentTransform.getRotationMatrix4x4();
    mat4x4 moveBackMat = translation4x4mat(totalTranslation);

    currentMatrix = moveBackMat * rotateMat * moveToCenterMat;
  }

  TRACYCZONEEND(matRotateVerts);

  // Map the PBO for CUDA access

  TRACYCZONENC(cudaMapPbo, "Cuda map PBO", true, PROFILER_DARK_GREEN);

  cudaGraphicsMapResources(1, &cudaPboResource, 0);

  uchar4 *pxlsPtr;
  size_t pxlsPtrSize;
  cudaGraphicsResourceGetMappedPointer((void **)&pxlsPtr, &pxlsPtrSize, cudaPboResource);

  TRACYCZONEEND(cudaMapPbo);

  // Camera setup

  TRACYCZONENC(cameraSettings, "Camera settings", true, PROFILER_TURQUOISE);

  updateCameraRaygenData(scene->cam);

  TRACYCZONEEND(cameraSettings);

  // Apply transforms to device objects

  computeDeviceTransforms();

  // Generate and trace rays

  rayTrace(pxlsPtr);

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

  FRAMEMARK;
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
    auto &cam = scene->cam;

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

  auto &cam = scene->cam;

  float3_L frontMovement = cam.camForward * sameDirMov;
  float3_L rightMovement = cam.camRight * rightDirMov;

  float3_L totalCamMov = normalize(frontMovement + rightMovement) * (movingSpeed * guiWindow.io->DeltaTime);
  cam.camPos += totalCamMov;
}