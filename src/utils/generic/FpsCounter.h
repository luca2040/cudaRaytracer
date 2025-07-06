#pragma once

#include <iostream>
#include <SDL2/SDL.h>

class FPScounter
{
private:
  int frameCount = 0;
  Uint32 lastTime = 0;

public:
  // To avoid errors while in the first second set to 60
  // It doesnt matter what value this is set to since it will get overwritten anyway.
  float fps = 60;

  inline void printFpsTag()
  {
    frameCount++;

    Uint32 currentTime = SDL_GetTicks();
    if (currentTime - lastTime >= 1000)
    {
      fps = frameCount * 1000.0f / (currentTime - lastTime);
      frameCount = 0;
      lastTime = currentTime;

      std::cout << "FPS: " << fps << std::endl;
    }
  }
};

extern FPScounter fpsCounter;