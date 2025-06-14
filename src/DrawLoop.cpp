#include <SDL2/SDL.h>

void keyPressed(SDL_Keycode key)
{
}

void drawFrame(SDL_Renderer *renderer, int WIDTH, int HEIGHT)
{
  Uint32 time = SDL_GetTicks();

  // Reset background
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  SDL_RenderClear(renderer);

  SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
  for (int x = 0; x < WIDTH; x++)
    for (int y = 0; y < HEIGHT; y++)
    {
      SDL_RenderDrawPoint(renderer, x, y);
    }

  SDL_RenderPresent(renderer);
}