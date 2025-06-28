#ifndef DRAW_LOOP
#define DRAW_LOOP

#include <SDL2/SDL.h>

const unsigned int WIDTH = 1400;
const unsigned int HEIGHT = 1400;

constexpr float HALF_WIDTH = static_cast<float>(WIDTH) * 0.5f;
constexpr float HALF_HEIGHT = static_cast<float>(HEIGHT) * 0.5f;

constexpr unsigned int TOTAL_PIXELS = WIDTH * HEIGHT;
constexpr float TWO_PI = 2.0f * M_PI;

void keyPressed(SDL_Keycode key);
void drawFrame(SDL_Renderer *renderer, SDL_Texture *texture);

#endif