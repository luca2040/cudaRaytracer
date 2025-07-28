#pragma once

#include <limits>
#include <cmath>
#include "../../math/Definitions.h"

// Axis-aligned bounding box
struct AABB
{
  float3_L l = {INFINITY, INFINITY, INFINITY};
  float3_L h = {-INFINITY, -INFINITY, -INFINITY};

  AABB() = default;
  AABB(float3_L l, float3_L h) : l(l), h(h) {}
};

inline void growBBtoInclude(AABB &box, float3_L point)
{
  box.l.x = fminf(point.x, box.l.x);
  box.l.y = fminf(point.y, box.l.y);
  box.l.z = fminf(point.z, box.l.z);

  box.h.x = fmaxf(point.x, box.h.x);
  box.h.y = fmaxf(point.y, box.h.y);
  box.h.z = fmaxf(point.z, box.h.z);
}