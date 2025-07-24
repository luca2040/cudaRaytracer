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
  float3_L *transformedPoints; // COPY OF "points"! Updated per frame
  triangleidx *triangles;
  size_t *dyntriangles;             // The indexes of all the triangles whose normals need to be recalculated
  transformIndexPair *trIndexPairs; // Basically start and end index of single objects in the vertex array
  SceneObject *sceneobjects;

  size_t pointsCount;
  size_t triangleNum;
  size_t dyntrianglesNum;
  size_t sceneobjectsNum;
  size_t trIndexPairCount;

  size_t pointsSize;
  size_t triangleSize;
  size_t sceneObjectsSize;

  // Device pointers
  // ONLY FOR CUDA
  float3_L *d_pointarray;
  triangleidx *d_triangles;
  SceneObject *d_sceneobjects;

  // Settings

  bool boundingBoxDebugView = false;
  int maxRayReflections = 5;

  Scene() = default;
};

extern Scene scene;