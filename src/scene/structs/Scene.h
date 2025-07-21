#pragma once

#include "Camera.h"
#include "../../scene/composition/SceneCompositor.h"

struct Scene
{
  // Camera

  Camera cam;

  // Triangles and stuff

  float3_L *points;
  triangleidx *triangles;
  size_t *dyntriangles; // The indexes of all the triangles whose normals need to be recalculated
  transformIndexPair *trIndexPairs; // Basically start and end index of single objects in the vertex array

  size_t pointsCount;
  size_t triangleNum;
  size_t dyntrianglesNum;
  size_t trIndexPairCount;

  size_t pointsSize;
  size_t triangleSize;

  Scene() = default;
};

extern Scene scene;