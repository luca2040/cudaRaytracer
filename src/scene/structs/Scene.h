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

  size_t pointsCount;
  size_t triangleNum;
  size_t trIndexPairCount;
  size_t sceneobjectsNum;
  // size_t matricesNum; // Same as sceneobjectsNum

  size_t pointsSize;
  size_t pointTableSize;
  size_t triangleSize;
  size_t sceneObjectsSize;
  size_t matricesSize;

  // Device pointers
  // ONLY FOR CUDA
  float3_L *d_pointarray;
  float3_L *d_trsfrmdpoints;
  size_t *d_pointToObjIdxTable;
  triangleidx *d_triangles;
  SceneObject *d_sceneobjects;
  mat4x4 *d_transformMatrices;

  // Settings

  bool transformSync = false;
  bool afterTraceSync = true;

  bool boundingBoxDebugView = false;
  int maxRayReflections = 15;

  Scene() = default;
};

extern Scene *scene;

extern Scene *d_scene;
extern size_t sceneStructSize;