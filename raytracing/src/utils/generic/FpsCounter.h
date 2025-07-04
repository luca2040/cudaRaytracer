#pragma once

#include <iostream>

// Global vars to keep count of FPS
int frameCount = 0;
Uint32 lastTime = 0;

float fps = 0;

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