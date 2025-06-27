#include <SDL2/SDL.h>
#include <iostream>

#include "../math/Definitions.h"
#include "DrawLoop.h"

#include "cuda/test.cuh"

// Camera settings
float camZ = 600;

// Rotations
float3 rotcenter(0.0f, 0.0f, 3000.0f);

float3 points[] = {
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
triangleidx triangles[] = {
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

void drawFrame(SDL_Renderer *renderer, SDL_Texture *texture)
{
  // Rotations

  constexpr float TWO_PI = 2.0f * M_PI;

  Uint32 time = SDL_GetTicks();
  float xrot = fmod((static_cast<float>(time) * 0.0005f), TWO_PI);
  float yrot = fmod((static_cast<float>(time) * 0.001f), TWO_PI);

  // Copy vertexes array

  constexpr size_t pointsCount = sizeof(points) / sizeof(points[0]);

  float3 pointarray[pointsCount];
  std::copy(std::begin(points), std::end(points), pointarray);

  mat3x3 yrotmat = {
      float3(cos(yrot), 0.0f, sin(yrot)),
      float3(0.0f, 1.0f, 0.0f),
      float3(-sin(yrot), 0.0f, cos(yrot))};

  mat3x3 xrotmat = {
      float3(1.0f, 0.0f, 0.0f),
      float3(0.0f, cos(xrot), -sin(xrot)),
      float3(0.0f, sin(xrot), cos(xrot))};

  mat3x3 rotCombined = xrotmat * yrotmat;

  for (size_t i = 0; i < pointsCount; i++)
  {

    // Vertex calculations and projection all compressed into a single cycle now

    pointarray[i] -= rotcenter;
    pointarray[i] = rotCombined * pointarray[i];
    pointarray[i] += rotcenter;

    float depthCamInverse = 1.0f / (camZ + pointarray[i].z);

    pointarray[i].x = pointarray[i].x * depthCamInverse * camZ + HALF_WIDTH;
    pointarray[i].y = pointarray[i].y * depthCamInverse * camZ + HALF_HEIGHT;
  }

  // Lock texture

  void *pixels;
  int texturePitch;
  SDL_LockTexture(texture, NULL, &pixels, &texturePitch);

  Uint32 *pixel_ptr = (Uint32 *)pixels;
  memset(pixel_ptr, 0, texturePitch * HEIGHT);

  // Draw borders

  constexpr size_t triangleNum = sizeof(triangles) / sizeof(triangles[0]);

  for (size_t i = 0; i < triangleNum; i++)
  {
    // The x and y here are the projected ones, the z is for the depth buffer

    triangleidx triangle = triangles[i];

    // Placeholder line implementation - can cause crashes

    auto drawLine = [](float2 point1, float2 point2, int color,
                       Uint32 *pixel_ptr, int texturePitch)
    {
      int x0 = static_cast<int>(point1.x);
      int y0 = static_cast<int>(point1.y);
      int x1 = static_cast<int>(point2.x);
      int y1 = static_cast<int>(point2.y);

      int dx = abs(x1 - x0);
      int dy = abs(y1 - y0);
      int sx = (x0 < x1) ? 1 : -1;
      int sy = (y0 < y1) ? 1 : -1;
      int err = dx - dy;

      while (true)
      {
        pixel_ptr[y0 * (texturePitch / 4) + x0] = color;

        if (x0 == x1 && y0 == y1)
          break;
        int e2 = 2 * err;
        if (e2 > -dy)
        {
          err -= dy;
          x0 += sx;
        }
        if (e2 < dx)
        {
          err += dx;
          y0 += sy;
        }
      }
    };

    drawLine(pointarray[triangle.v1], pointarray[triangle.v2], triangle.col, pixel_ptr, texturePitch);
    drawLine(pointarray[triangle.v2], pointarray[triangle.v3], triangle.col, pixel_ptr, texturePitch);
    drawLine(pointarray[triangle.v1], pointarray[triangle.v3], triangle.col, pixel_ptr, texturePitch);
  }

  // Unlock and render texture

  SDL_UnlockTexture(texture);
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, NULL, NULL);
}

void keyPressed(SDL_Keycode key)
{
  testCUDA();
}