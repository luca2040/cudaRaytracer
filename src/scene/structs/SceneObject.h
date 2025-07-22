#pragma once

#include "../../math/Definitions.h"
#include "AABB.h"

struct SceneObject
{
  size_t triangleStartIdx;
  size_t triangleEndIdx;

  AABB boundingBox;

  SceneObject() = default;
  SceneObject(size_t triangleStartIdx, size_t triangleEndIdx, AABB boundingBox)
      : triangleStartIdx(triangleStartIdx),
        triangleEndIdx(triangleEndIdx),
        boundingBox(boundingBox) {}
};