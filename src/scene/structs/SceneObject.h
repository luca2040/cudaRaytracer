#pragma once

#include "../../math/Definitions.h"
#include "AABB.h"
#include "Sphere.h"

struct SceneObject
{
  size_t triangleStartIdx;
  size_t triangleNum;

  size_t vertStartIdx;
  size_t vertNum;

  // size_t matrixIdx; // Dont need, mat idx is the same as the object's one

  AABB boundingBox;
  Sphere sphere;

  SceneObject() = default;
  SceneObject(size_t triangleStartIdx, size_t triangleNum, // Mesh
              size_t vertStartIdx, size_t vertNum,
              // size_t matrixIdx,
              AABB boundingBox)
      : triangleStartIdx(triangleStartIdx),
        triangleNum(triangleNum),
        vertStartIdx(vertStartIdx),
        vertNum(vertNum),
        // matrixIdx(matrixIdx),
        boundingBox(boundingBox)
  {
  }
  SceneObject(Sphere sphere) : sphere(sphere) // Sphere
  {
  }
};