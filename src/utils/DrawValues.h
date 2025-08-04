#pragma once

#include "../math/Definitions.h"
#include <math.h>

// #define EPSILON 1.1920929e-05f
#define EPSILON 0.001f

#define RAYTRACE_BLOCK_SIDE 16
#define VERT_THREADS_PER_BLOCK 256

#define TWO_PI 6.28318530717958647692

// Angle to remove from camera vertical rotation both sides, in degrees
const float cameraVerticalViewReduction = 10.0f;

const float cameraVerticalMaxRot = M_PI_2 - (cameraVerticalViewReduction / 180.0f * M_PI);
const float cameraVerticalMinRot = -cameraVerticalMaxRot;