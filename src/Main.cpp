#include <iostream>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <glm/glm.hpp>

#include "utils/DrawLoop.h"
#include "utils/FpsCounter.h"

int main()
{
  std::cout << "Starting the best 3D renderer ever!" << std::endl;

  if (SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    std::cerr << "SDL could not initialize: " << SDL_GetError() << std::endl;
    return 1;
  }
  if (TTF_Init() < 0)
  {
    std::cout << "Couldn't initialize TTF lib: " << TTF_GetError() << std::endl;
    return 1;
  }

  SDL_Window *window = SDL_CreateWindow("The Best 3D Renderer Ever",
                                        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                        WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
  SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED); // | SDL_RENDERER_PRESENTVSYNC);
  SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888,
                                           SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

  TTF_Font *font = TTF_OpenFont("./font.ttf", 24);
  if (!font)
  {
    SDL_Log("Font load error: %s", TTF_GetError());
    return 1;
  }

  SDL_Event event;
  bool running = true;

  glm::ivec2 mouse(0, 0);
  glm::ivec2 pMouse(0, 0);

  while (running)
  {
    while (SDL_PollEvent(&event))
    {
      switch (event.type)
      {
      case SDL_QUIT:
        running = false;
        break;

      case SDL_KEYDOWN:
        std::cout << "Key pressed: " << SDL_GetKeyName(event.key.keysym.sym) << std::endl;

        if (event.key.keysym.sym == SDLK_ESCAPE)
          running = false;

        keyPressed(event.key.keysym.sym);

        break;

      case SDL_MOUSEMOTION:
        pMouse = mouse;
        mouse = glm::ivec2(event.motion.x, event.motion.y);
        break;

      case SDL_MOUSEBUTTONDOWN:
        // Not implemented right now - still dont need it
        std::cout << "Mouse button " << (int)event.button.button
                  << " pressed at (" << event.button.x << ", " << event.button.y << ")" << std::endl;
        break;
      }
    }

    // Main draw logic
    drawFrame(renderer, texture);

    // Draw frame counter
    renderFpsTag(renderer, font);

    // Actually print to screen
    SDL_RenderPresent(renderer);
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}