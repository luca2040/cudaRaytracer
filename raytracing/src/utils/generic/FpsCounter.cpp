#include <sstream>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

SDL_Color textColor = {255, 255, 255, 255};

// Global vars to keep count of FPS
int frameCount = 0;
Uint32 lastTime = 0;

float fps = 0;

void renderFpsTag(SDL_Renderer *renderer, TTF_Font *font)
{
  std::ostringstream ss;
  ss << "FPS: " << fps;
  SDL_Surface *textSurface = TTF_RenderText_Blended(font, ss.str().c_str(), textColor);
  SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

  SDL_Rect dstRect = {10, 10, textSurface->w, textSurface->h};
  SDL_RenderCopy(renderer, textTexture, NULL, &dstRect);

  SDL_FreeSurface(textSurface);
  SDL_DestroyTexture(textTexture);

  frameCount++;

  Uint32 currentTime = SDL_GetTicks();
  if (currentTime - lastTime >= 1000)
  {
    fps = frameCount * 1000.0f / (currentTime - lastTime);
    frameCount = 0;
    lastTime = currentTime;
  }
}