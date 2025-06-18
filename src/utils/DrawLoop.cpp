#include <SDL2/SDL.h>
#include <iostream>

#include "../math/mathUtils.h"
#include "DrawingUtils.h"

// Camera settings
float camZ = 600;

// Rotations
float rotcenter[3] = {0.0f, 0.0f, 3000.0f};
float yrot = 0;
float xrot = 0;

float points[8][3] = {
    // Front face
    {-1000.0f, -1000.0f, 2000.0f},
    {1000.0f, -1000.0f, 2000.0f},
    {1000.0f, 1000.0f, 2000.0f},
    {-1000.0f, 1000.0f, 2000.0f},
    // Back face
    {-1000.0f, -1000.0f, 4000.0f},
    {1000.0f, -1000.0f, 4000.0f},
    {1000.0f, 1000.0f, 4000.0f},
    {-1000.0f, 1000.0f, 4000.0f}};

// Vertex intexes [3], color
int triangles[12][4] = {
    {0, 1, 3, 0xFF0000},
    {1, 3, 2, 0xFF0000},

    {1, 5, 2, 0x00FF00},
    {5, 2, 6, 0x00FF00},

    {0, 3, 4, 0x0000FF},
    {4, 3, 7, 0x0000FF},

    {4, 7, 5, 0xFFFF00},
    {5, 7, 6, 0xFFFF00},

    {0, 1, 4, 0xFF00FF},
    {1, 4, 5, 0xFF00FF},

    {3, 2, 7, 0x00FFFF},
    {2, 7, 6, 0x00FFFF},
};

void drawFrame(SDL_Renderer *renderer, SDL_Texture *texture, int WIDTH, int HEIGHT)
{
  int TOTAL_PIXEL_NUM = WIDTH * HEIGHT;

  // Rotations

  Uint32 time = SDL_GetTicks();
  xrot = fmod((float(time) / 2000.0f), (2.0f * M_PI));
  yrot = fmod((float(time) / 1000.0f), (2.0f * M_PI));

  // Copy vertexes array

  float pointarray[8][3];

  size_t pointsRows = sizeof(points) / sizeof(points[0]);

  for (size_t i = 0; i < pointsRows; i++)
    for (size_t j = 0; j < 3; j++)
      pointarray[i][j] = points[i][j];

  // Subtract center to perform rotation around that

  sumArrays(pointarray, pointsRows, rotcenter, -1.0f);

  // Apply rotations around that point

  float yrotmat[3][3] = {
      {cos(yrot), 0.0f, sin(yrot)},
      {0.0f, 1.0f, 0.0f},
      {-sin(yrot), 0.0f, cos(yrot)}};

  float xrotmat[3][3] = {
      {1.0f, 0.0f, 0.0f},
      {0.0f, cos(xrot), -sin(xrot)},
      {0.0f, sin(xrot), cos(xrot)}};

  for (size_t i = 0; i < pointsRows; i++)
  {
    matMult(pointarray[i], yrotmat);
    matMult(pointarray[i], xrotmat);
  }

  // Re-add center to remove the offset keeping the rotation

  sumArrays(pointarray, pointsRows, rotcenter, 1.0f);

  // Calc and draw points

  float points2d[pointsRows][3];

  for (size_t i = 0; i < pointsRows; i++)
  {
    float xr = (camZ * pointarray[i][0]) / (camZ + pointarray[i][2]);
    float yr = (camZ * pointarray[i][1]) / (camZ + pointarray[i][2]);

    // Shift for the window, because it assumes 0,0 is top left, not center
    // Not important for the calculations [TODO] optimize this thing
    xr += static_cast<float>(WIDTH / 2);
    yr += static_cast<float>(HEIGHT / 2);

    points2d[i][0] = xr;
    points2d[i][1] = yr;
    points2d[i][2] = pointarray[i][2];
  }

  // Lock texture

  void *pixels;
  int texturePitch;
  SDL_LockTexture(texture, NULL, &pixels, &texturePitch);

  Uint32 *pixel_ptr = (Uint32 *)pixels;
  memset(pixel_ptr, 0, texturePitch * HEIGHT);

  // Init "drawing depth buffer" - idk how to call this abomination of an idea

  float drawDepthBuffer[TOTAL_PIXEL_NUM];
  std::fill_n(drawDepthBuffer, TOTAL_PIXEL_NUM, std::nanf(""));

  // To use this: drawDepthBuffer[y * WIDTH + x]

  // Draw borders

  size_t triangleNum = sizeof(triangles) / sizeof(triangles[0]);

  for (size_t i = 0; i < triangleNum; i++)
  {
    // The x and y are already projected, the z is just to get priority on drawing

    float x1 = points2d[triangles[i][0]][0];
    float y1 = points2d[triangles[i][0]][1];
    float z1 = points2d[triangles[i][0]][2];

    float x2 = points2d[triangles[i][1]][0];
    float y2 = points2d[triangles[i][1]][1];
    float z2 = points2d[triangles[i][1]][2];

    float x3 = points2d[triangles[i][2]][0];
    float y3 = points2d[triangles[i][2]][1];
    float z3 = points2d[triangles[i][2]][2];

    int color = triangles[i][3];

    // SDL_SetRenderDrawColor(renderer, red, green, blue, 255);

    // Render it

    // SDL_RenderDrawLine(renderer, static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2), static_cast<int>(y2));
    // SDL_RenderDrawLine(renderer, static_cast<int>(x2), static_cast<int>(y2), static_cast<int>(x3), static_cast<int>(y3));
    // SDL_RenderDrawLine(renderer, static_cast<int>(x3), static_cast<int>(y3), static_cast<int>(x1), static_cast<int>(y1));

    depthFillTriangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, pixel_ptr, texturePitch, drawDepthBuffer, color, WIDTH, HEIGHT);
  }

  // Unlock and render texture

  SDL_UnlockTexture(texture);
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, NULL, NULL);
}

void keyPressed(SDL_Keycode key)
{
}