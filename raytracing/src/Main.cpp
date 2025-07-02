#include <iostream>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <GL/glew.h>

#include "utils/DrawLoop.h"
#include "utils/generic/FpsCounter.h"
#include "math/Definitions.h"

#include "third_party/tracy/tracy/Tracy.hpp"

int main()
{
  std::cout << "Starting the best 3D renderer ever - Now RTX!" << std::endl;

  if (SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    std::cerr << "SDL could not initialize: " << SDL_GetError() << std::endl;
    return 1;
  }
  if (TTF_Init() < 0)
  {
    std::cerr << "Couldn't initialize TTF lib: " << TTF_GetError() << std::endl;
    return 1;
  }

  SDL_Window *window = SDL_CreateWindow("The Best 3D Renderer Ever - RTX edition",
                                        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                        WIDTH, HEIGHT, SDL_WINDOW_SHOWN);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  SDL_GLContext glContext = SDL_GL_CreateContext(window);

  if (glewInit() != GLEW_OK)
  {
    std::cerr << "GLEW failed to initialize" << std::endl;
    return -1;
  }

  glViewport(0, 0, WIDTH, HEIGHT);

  GLuint pbo, tex;

  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, nullptr, GL_STREAM_DRAW);

  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  TTF_Font *font = TTF_OpenFont("./font.ttf", 24);
  if (!font)
  {
    SDL_Log("Font load error: %s", TTF_GetError());
    return 1;
  }

  SDL_Event event;
  bool running = true;

  int2_L mouse(0, 0);
  int2_L pMouse(0, 0);

  onSceneComposition();
  onSetupFrame(pbo);

  while (running)
  {
    ZoneScopedN("Main SDL while cycle");

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
        mouse = int2_L(event.motion.x, event.motion.y);
        break;

      case SDL_MOUSEBUTTONDOWN:
        // Not implemented right now - still dont need it
        std::cout << "Mouse button " << (int)event.button.button
                  << " pressed at (" << event.button.x << ", " << event.button.y << ")" << std::endl;
        break;
      }
    }

    // Main draw logic
    drawFrame(tex, pbo);

    // Draw frame counter
    // renderFpsTag(renderer, font);

    // Actually print to screen
    // SDL_RenderPresent(renderer);
  }

  onClose();

  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}