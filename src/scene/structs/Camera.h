#pragma once

#include "../../math/Definitions.h"

struct Camera
{
  // Camera vectors

  float3_L camForward = {0, 0, 0};
  float3_L camRight = {0, 0, 0};
  float3_L camUp = {0, 0, 0};

  // Camera

  float camFOVdeg = 75;
  float camZoom = 2.0f;

  float3_L camPos = {0, 0, 0};

  float camXrot = 0;
  float camYrot = 0;

  Camera() = default;
};