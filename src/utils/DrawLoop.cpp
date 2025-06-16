#include <SDL2/SDL.h>
#include <iostream>

#include "../math/mathUtils.h"

// Camera settings
float camZ = 600;

// Rotations
float rotcenter[3] = {0.0f, 0.0f, 3000.0f};
float yrot = 0;
float xrot = 0;

float pointscube[8][3] = {
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

int connectionscube[12][2] = {
    // Front face
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 0},
    // Back face
    {4, 5},
    {5, 6},
    {6, 7},
    {7, 4},
    // Mid connections
    {0, 4},
    {1, 5},
    {2, 6},
    {3, 7}};

void drawFrame(SDL_Renderer *renderer, int WIDTH, int HEIGHT)
{
  Uint32 time = SDL_GetTicks();

  // Reset background
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  SDL_RenderClear(renderer);
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

  // Rotations

  xrot = fmod((float(time) / 2000.0f), (2.0f * M_PI));
  yrot = fmod((float(time) / 1000.0f), (2.0f * M_PI));

  // Copy vertexes array

  float pointarray[8][3];

  size_t pointsRows = sizeof(pointscube) / sizeof(pointscube[0]);

  for (size_t i = 0; i < pointsRows; i++)
    for (size_t j = 0; j < 3; j++)
      pointarray[i][j] = pointscube[i][j];

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

  float points2d[pointsRows][2];

  for (size_t i = 0; i < pointsRows; i++)
  {
    float xr = (camZ * pointarray[i][0]) / (camZ + pointarray[i][2]);
    float yr = (camZ * pointarray[i][1]) / (camZ + pointarray[i][2]);

    // Shift for the window, because it assumes 0,0 is top left, not center
    // Not important for the calculations
    xr += static_cast<float>(WIDTH) / 2.0f;
    yr += static_cast<float>(HEIGHT) / 2.0f;

    points2d[i][0] = xr;
    points2d[i][1] = yr;
  }

  // Draw borders

  size_t connectionsRows = sizeof(connectionscube) / sizeof(connectionscube[0]);

  for (size_t i = 0; i < connectionsRows; i++)
  {
    float x1 = points2d[connectionscube[i][0]][0];
    float y1 = points2d[connectionscube[i][0]][1];

    float x2 = points2d[connectionscube[i][1]][0];
    float y2 = points2d[connectionscube[i][1]][1];

    SDL_RenderDrawLine(renderer, static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2), static_cast<int>(y2));
  }
}

void keyPressed(SDL_Keycode key)
{
  // Testing

  float result = triangleInterpolate(0, 0, 10, 0, 5, 10, 6, 8, 1, 2, 3);

  std::cout << result << std::endl;
}