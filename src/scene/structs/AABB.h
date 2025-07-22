#pragma once

#include "../../math/Definitions.h"

// Axis-aligned bounding box
struct AABB
{
  float3_L l;
  float3_L h;

  AABB() = default;
  AABB(float3_L l, float3_L h) : l(l), h(h) {}
};

inline void setBBtoNew(AABB &box)
{
  box.l.x = INFINITY; // I use this like a keyword
}

inline bool isBBnew(const AABB &box)
{
  return box.l.x == INFINITY;
}

inline void growBBtoInclude(AABB &box, float3_L point)
{
  if (isBBnew(box))
  {
    box.l = point;
    box.h = point;
    return;
  }

  box.l.x = fminf(point.x, box.l.x);
  box.l.y = fminf(point.y, box.l.y);
  box.l.z = fminf(point.z, box.l.z);

  box.h.x = fmaxf(point.x, box.h.x);
  box.h.y = fmaxf(point.y, box.h.y);
  box.h.z = fmaxf(point.z, box.h.z);
}