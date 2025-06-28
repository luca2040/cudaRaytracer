#include <SDL2/SDL.h>
#include <iostream>

#include "../math/Definitions.h"
#include "../math/Operations.h"
#include "DrawLoop.h"

#include "cuda/test.cuh"

// Camera settings
float camZ = 600;
float camGridSize = 1;
float3 camPos = {0, 0, 0};
float3 camRotation = {0, 0, M_PI};

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

  Uint32 time = SDL_GetTicks();
  float xrot = fmod((static_cast<float>(time) * 0.0005f), TWO_PI);
  float yrot = fmod((static_cast<float>(time) * 0.001f), TWO_PI);

  // Copy vertexes array

  constexpr size_t pointsCount = sizeof(points) / sizeof(points[0]);

  float3 *pointarray = new float3[pointsCount];
  std::copy(std::begin(points), std::end(points), pointarray);

  // Apply rotations to copied list

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
    pointarray[i] -= rotcenter;
    pointarray[i] = rotCombined * pointarray[i];
    pointarray[i] += rotcenter;
  }

  // Lock texture

  void *pixels;
  int texturePitch;
  SDL_LockTexture(texture, NULL, &pixels, &texturePitch);

  Uint32 *pixel_ptr = (Uint32 *)pixels;
  memset(pixel_ptr, 0, texturePitch * HEIGHT);

  // Generate rays

  ray *rays = new ray[TOTAL_PIXELS];
  // rays[y * WIDTH + x]

  // Camera rotation

  mat3x3 xCamRotmat = {
      float3(1.0f, 0.0f, 0.0f),
      float3(0.0f, cos(camRotation.x), -sin(camRotation.x)),
      float3(0.0f, sin(camRotation.x), cos(camRotation.x))};

  mat3x3 yCamRotmat = {
      float3(cos(camRotation.y), 0.0f, sin(camRotation.y)),
      float3(0.0f, 1.0f, 0.0f),
      float3(-sin(camRotation.y), 0.0f, cos(camRotation.y))};

  mat3x3 zCamRotmat = {
      float3(cos(camRotation.z), -sin(camRotation.z), 0.0f),
      float3(sin(camRotation.z), cos(camRotation.z), 0.0f),
      float3(0.0f, 0.0f, 1.0f)};

  mat3x3 rotCamCombined = xCamRotmat * yCamRotmat * zCamRotmat;

  for (int y = 0; y < HEIGHT; y++)
  {
    for (int x = 0; x < WIDTH; x++)
    {
      float xGridCoord = (x - HALF_WIDTH + 0.5f) * camGridSize;
      float yGridCoord = (y - HALF_HEIGHT + 0.5f) * camGridSize;

      // Vector to xGridCoord, yGridCoord, camZ
      float3 normalizedDir = normalize({xGridCoord, yGridCoord, camZ});
      normalizedDir = rotCamCombined * normalizedDir;

      rays[y * WIDTH + x] = ray(camPos, normalizedDir);
    }
  }

  // Trace the rays

  constexpr size_t triangleNum = sizeof(triangles) / sizeof(triangles[0]);

  for (int y = 0; y < HEIGHT; y++)
  {
    for (int x = 0; x < WIDTH; x++)
    {
      ray currentRay = rays[y * WIDTH + x];

      // Bruteforce all the triangles
      float currentZbuf = INFINITY;
      for (size_t i = 0; i < triangleNum; i++)
      {
        triangleidx triangle = triangles[i];

        float3 v1 = pointarray[triangle.v1];
        float3 v2 = pointarray[triangle.v2];
        float3 v3 = pointarray[triangle.v3];
        unsigned int color = triangle.col;

        std::optional<float3> intersectionPoint = ray_intersects_triangle(currentRay.origin, currentRay.direction, v1, v2, v3);
        if (intersectionPoint)
        {
          float3 intrstPoint = intersectionPoint.value();

          float distanceToCamera = distance(camPos, intrstPoint);
          if (distanceToCamera < currentZbuf)
          {
            currentZbuf = distanceToCamera;
            pixel_ptr[y * (texturePitch / 4) + x] = color;
          }
        }
      }
    }
  }

  // pixel_ptr[y0 * (texturePitch / 4) + x0] = color;

  // for (size_t i = 0; i < triangleNum; i++)
  // {
  //   // The x and y here are the projected ones, the z is for the depth buffer

  //   // triangleidx triangle = triangles[i];
  //   // pointarray[triangle.v1]
  // }

  // Clean up

  delete[] rays;
  delete[] pointarray;

  // Unlock and render texture

  SDL_UnlockTexture(texture);
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, NULL, NULL);
}

void keyPressed(SDL_Keycode key)
{
  testCUDA();
}