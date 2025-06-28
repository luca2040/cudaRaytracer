#ifndef DRAW_LOOP
#define DRAW_LOOP

#include <SDL2/SDL.h>
#include "DrawValues.h"

void keyPressed(SDL_Keycode key);
void onSetupFrame(SDL_Renderer *renderer, SDL_Texture *texture);
void onClose(SDL_Renderer *renderer, SDL_Texture *texture);
void drawFrame(SDL_Renderer *renderer, SDL_Texture *texture);

#endif