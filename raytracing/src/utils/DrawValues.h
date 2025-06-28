#ifndef DRAW_VALUES
#define DRAW_VALUES

#include <math.h>

const unsigned int WIDTH = 1400;
const unsigned int HEIGHT = 1400;

constexpr float HALF_WIDTH = static_cast<float>(WIDTH) * 0.5f;
constexpr float HALF_HEIGHT = static_cast<float>(HEIGHT) * 0.5f;

constexpr float ASPECT = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);

constexpr unsigned int TOTAL_PIXELS = WIDTH * HEIGHT;
constexpr float TWO_PI = 2.0f * M_PI;

#endif