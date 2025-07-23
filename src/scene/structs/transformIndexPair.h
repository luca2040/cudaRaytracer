#pragma once

#include "../../math/Definitions.h"
#include "../SceneClasses.h"

struct transformIndexPair
{
  size_t startIdx;
  size_t endIdx;
  ObjTransform transform;

  inline transformIndexPair() = default;
  inline transformIndexPair(size_t start, size_t end, const ObjTransform &transform)
      : startIdx(start), endIdx(end), transform(transform) {}
};