#pragma once

#include "../math/Definitions.h"
#include <math.h>

const unsigned int WIDTH = 3840;
const unsigned int HEIGHT = 2160;

const int BG_COLOR = 0x000000;

constexpr float HALF_WIDTH = static_cast<float>(WIDTH) * 0.5f;
constexpr float HALF_HEIGHT = static_cast<float>(HEIGHT) * 0.5f;

constexpr float ASPECT = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);

constexpr unsigned int TOTAL_PIXELS = WIDTH * HEIGHT;
constexpr float TWO_PI = 2.0f * M_PI;

// Angle to remove from camera vertical rotation both sides, in degrees
constexpr float cameraVerticalViewReduction = 10.0f;

constexpr float cameraVerticalMaxRot = M_PI_2 - (cameraVerticalViewReduction / 180.0f * M_PI);
constexpr float cameraVerticalMinRot = -cameraVerticalMaxRot;

#define EPSILON 1.1920929e-05f

// How many times rays should be traced, 2 means an hit and single reflection
#define RAY_HITS_MAX 10