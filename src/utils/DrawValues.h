#pragma once

#include "../math/Definitions.h"
#include <math.h>

// ########### Window size config ###########

#define windowWidth 3840
#define windowHeight 2160

#define renderingReduction 2

// ############ Rendering config ############

#define EPSILON 1.1920929e-05f
const int BG_COLOR = 0x224422;

#define RAYTRACE_BLOCK_SIDE 16
#define VERT_THREADS_PER_BLOCK 256

// ##########################################

const unsigned int WINDOW_WIDTH = windowWidth;
const unsigned int WINDOW_HEIGHT = windowHeight;

#ifdef renderingReduction

const unsigned int WIDTH = WINDOW_WIDTH / renderingReduction;
const unsigned int HEIGHT = WINDOW_HEIGHT / renderingReduction;

#else

const unsigned int WIDTH = WINDOW_WIDTH;
const unsigned int HEIGHT = WINDOW_HEIGHT;

#endif

constexpr float HALF_WIDTH = static_cast<float>(WIDTH) * 0.5f;
constexpr float HALF_HEIGHT = static_cast<float>(HEIGHT) * 0.5f;

constexpr float ASPECT = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);

constexpr unsigned int TOTAL_PIXELS = WIDTH * HEIGHT;
constexpr float TWO_PI = 2.0f * M_PI;

// Angle to remove from camera vertical rotation both sides, in degrees
constexpr float cameraVerticalViewReduction = 10.0f;

constexpr float cameraVerticalMaxRot = M_PI_2 - (cameraVerticalViewReduction / 180.0f * M_PI);
constexpr float cameraVerticalMinRot = -cameraVerticalMaxRot;