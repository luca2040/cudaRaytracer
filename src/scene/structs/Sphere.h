#pragma once

#include "../../math/Definitions.h"

struct Sphere
{
  bool isValid = false; // Because all objects have a Sphere instance but not all of 'em actually are
  float3_L center;
  float radius;

  size_t materialIdx;

  Sphere() = default;
  Sphere(float3_L center, float radius) : center(center),
                                          radius(radius)
  {
    isValid = true;
  }
};
