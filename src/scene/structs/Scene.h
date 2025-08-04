#pragma once

#include "Camera.h"
#include "SceneObject.h"
#include "transformIndexPair.h"

struct Scene
{
  // Camera

  Camera cam;

  // Triangles and stuff

  float3_L *points;
  size_t *pointToObjIdxTable;
  triangleidx *triangles;
  transformIndexPair *trIndexPairs; // Reference to sceneobject that also contains transform functions
  SceneObject *sceneobjects;
  mat4x4 *transformMatrices;
  Material *materials;

  size_t pointsCount;
  size_t triangleNum;
  size_t trIndexPairCount;
  size_t sceneobjectsNum;
  // size_t matricesNum; // Same as sceneobjectsNum
  size_t materialsNum;

  size_t pointsSize;
  size_t pointTableSize;
  size_t triangleSize;
  size_t sceneObjectsSize;
  size_t matricesSize;
  size_t materialsSize;

  float accumulatedFrames;

  // Device pointers
  // ONLY FOR CUDA
  float3_L *d_accumulationBuffer;

  float3_L *d_pointarray;
  float3_L *d_trsfrmdpoints;
  size_t *d_pointToObjIdxTable;
  triangleidx *d_triangles;
  SceneObject *d_sceneobjects;
  mat4x4 *d_transformMatrices;
  Material *d_materials;

  // Debug settings

  bool transformSync = false;
  bool afterTraceSync = true;
  bool boundingBoxDebugView = false;

  // Rendering settings

  bool motionPause = false;
  bool simpleRender = true;
  bool randomize = false;
  bool accumulate = false;

  float3_L backgroundColor = {0.4f, 0.5f, 0.9f};
  int samplesPerPixel = 20;
  float pixelSampleRange = 0.5; // In the same unit as pixels - 0.5 -> half pixel
  int maxRayReflections = 4;

  Scene() = default;
};

extern Scene *scene;

extern Scene *d_scene;
extern size_t sceneStructSize;