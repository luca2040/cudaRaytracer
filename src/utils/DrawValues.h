#pragma once

#include "../math/Definitions.h"
#include <math.h>

// ############ Rendering config ############

#define EPSILON 1.1920929e-05f

#define RAYTRACE_BLOCK_SIDE 16
#define VERT_THREADS_PER_BLOCK 256

// ##########################################

constexpr float TWO_PI = 2.0f * M_PI;

// Angle to remove from camera vertical rotation both sides, in degrees
constexpr float cameraVerticalViewReduction = 10.0f;

constexpr float cameraVerticalMaxRot = M_PI_2 - (cameraVerticalViewReduction / 180.0f * M_PI);
constexpr float cameraVerticalMinRot = -cameraVerticalMaxRot;