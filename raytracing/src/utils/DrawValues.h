#pragma once

#include <math.h>

const unsigned int WIDTH = 1400;
const unsigned int HEIGHT = 1400;

const unsigned int BG_COLOR = 0x000000;

constexpr float HALF_WIDTH = static_cast<float>(WIDTH) * 0.5f;
constexpr float HALF_HEIGHT = static_cast<float>(HEIGHT) * 0.5f;

constexpr float ASPECT = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);

constexpr unsigned int TOTAL_PIXELS = WIDTH * HEIGHT;
constexpr float TWO_PI = 2.0f * M_PI;

#define EPSILON 1.1920929e-07f

struct DrawingLoopValues
{
  size_t simpleCubeIndex;
  size_t movingCubeIndex;
};