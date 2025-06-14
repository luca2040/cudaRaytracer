#include <iostream>
#include <SDL2/SDL.h>

const int width = 1080;
const int height = 1080;

int main()
{
  std::cout << "Starting the best 3D renderer ever!";

  if (SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    std::cerr << "SDL could not initialize: " << SDL_GetError() << std::endl;
    return 1;
  }

  return 0;
}