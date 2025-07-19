#pragma once

#include "Camera.h"

struct Scene
{
  Camera cam;

  Scene() = default;
};

extern Scene scene;