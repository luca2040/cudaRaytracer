#include <SDL2/SDL.h>
#include <iostream>

#include "../math/Definitions.h"
#include "../math/Operations.h"
#include "DrawLoop.h"

#include "cuda/test.cuh"

// ######################### Camera settings ##########################

const float camFOVdeg = 90;

float3 camPos = {0, 0, 0};
float3 camLookingPoint = {0, 0, 1};

// ####################################################################

const float camFOV = camFOVdeg * M_PI / 180.0f;
const float imagePlaneHeight = 2.0f * tan(camFOV / 2.0f);
const float imagePlaneWidth = imagePlaneHeight * ASPECT;

// ####################################################################

// Objects
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
  float xrot = fmod((static_cast<float>(time) * 0.00005f), TWO_PI);
  float yrot = fmod((static_cast<float>(time) * 0.0001f), TWO_PI);

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

  // Camera setup

  float3 camForward = normalize(camLookingPoint - camPos);
  float3 camRight = normalize(cross3(camForward, {0, -1, 0})); // camForward, worldUp
  float3 camUp = cross3(camRight, camForward);

  // Camera placement

  float3 imageCenter = camPos + camForward;
  float3 imageX = camRight * imagePlaneWidth;
  float3 imageY = camUp * imagePlaneHeight;
  float3 camViewOrigin = imageCenter - imageX * 0.5f - imageY * 0.5f;

  for (int y = 0; y < HEIGHT; y++)
  {
    for (int x = 0; x < WIDTH; x++)
    {
      // [TODO] take out the divisions and add option for zoom

      float u = static_cast<float>(x) / static_cast<float>(WIDTH - 1);
      float v = static_cast<float>(y) / static_cast<float>(HEIGHT - 1);

      float3 pixelPos = camViewOrigin + imageX * u + imageY * v;
      float3 rayDir = pixelPos - camPos;

      rays[y * WIDTH + x] = ray(camPos, rayDir);
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