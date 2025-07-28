#pragma once

#include "../../math/Definitions.h"
#include "../SceneClasses.h"

struct transformIndexPair
{
  ObjTransform transform;
  size_t sceneObjectReference;

  inline transformIndexPair() = default;
  inline transformIndexPair(const ObjTransform &transform, size_t sceneObjectReference)
      : transform(transform), sceneObjectReference(sceneObjectReference) {}
};