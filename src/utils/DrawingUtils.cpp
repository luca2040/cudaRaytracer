#include <algorithm>
#include <SDL2/SDL.h>

#include "../math/mathUtils.h"
#include "../math/names.h"
#include "DrawLoop.h"

void drawHorizontalLine(int xStart, int xEnd, int lineY,
                        float3 t_v1,
                        float3 t_v2,
                        float3 t_v3,
                        Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
                        int color,
                        float2 &b_v0, float2 &b_v1, float &b_d00, float &b_d01, float &b_d11, float &b_invDenom)
{
  if (lineY > HEIGHT || lineY < 0)
    return;

  if (xStart > xEnd)
    std::swap(xStart, xEnd);

  for (int pointX = xStart; pointX <= xEnd; pointX++)
  {
    if (pointX > WIDTH || pointX < 0)
      continue;

    float pointZ = triangleInterpolate(t_v1, t_v2, t_v3,
                                       pointX, lineY,
                                       b_v0, b_v1, b_d00, b_d01, b_d11, b_invDenom);
    float &actualZ = drawDepthBuffer[lineY * WIDTH + pointX];

    if (actualZ < pointZ)
      continue;
    actualZ = pointZ;

    pixel_ptr[lineY * (texturePitch / 4) + pointX] = color;
  }
}

void fillBottomFlatTriangle(
    float3 t_v1,
    float3 t_v2,
    float3 t_v3,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3,
    Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
    int color,
    float2 &b_v0, float2 &b_v1, float &b_d00, float &b_d01, float &b_d11, float &b_invDenom)
{
  float invslope1 = (x2 - x1) / (y2 - y1);
  float invslope2 = (x3 - x1) / (y3 - y1);

  float curx1 = x1;
  float curx2 = x1;

  for (int scanlineY = y1; scanlineY <= y2; scanlineY++)
  {
    drawHorizontalLine(static_cast<int>(curx1), static_cast<int>(curx2), scanlineY,
                       t_v1, t_v2, t_v3,
                       pixel_ptr, texturePitch, drawDepthBuffer,
                       color,
                       b_v0, b_v1, b_d00, b_d01, b_d11, b_invDenom);

    curx1 += invslope1;
    curx2 += invslope2;
  }
}

void fillTopFlatTriangle(
    float3 t_v1,
    float3 t_v2,
    float3 t_v3,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3,
    Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
    int color,
    float2 &b_v0, float2 &b_v1, float &b_d00, float &b_d01, float &b_d11, float &b_invDenom)
{
  float invslope1 = (x3 - x1) / (y3 - y1);
  float invslope2 = (x3 - x2) / (y3 - y2);

  float curx1 = x3;
  float curx2 = x3;

  for (int scanlineY = y3; scanlineY > y1; scanlineY--)
  {
    drawHorizontalLine(static_cast<int>(curx1), static_cast<int>(curx2), scanlineY,
                       t_v1, t_v2, t_v3,
                       pixel_ptr, texturePitch, drawDepthBuffer,
                       color,
                       b_v0, b_v1, b_d00, b_d01, b_d11, b_invDenom);

    curx1 -= invslope1;
    curx2 -= invslope2;
  }
}

void rasterizeFullTriangle(
    float3 v1,
    float3 v2,
    float3 v3,
    Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
    int color)
{

  // Sort vertices by y

  if (v1.y > v2.y)
    std::swap(v1, v2);
  if (v2.y > v3.y)
    std::swap(v2, v3);
  if (v1.y > v2.y)
    std::swap(v1, v2);

  // Pre-calc values for barycentric coordinates

  float2 b_v0 = v2 - v1, b_v1 = v3 - v1;
  float b_d00 = dot(b_v0, b_v0);
  float b_d01 = dot(b_v0, b_v1);
  float b_d11 = dot(b_v1, b_v1);
  float b_invDenom = 1.0 / (b_d00 * b_d11 - b_d01 * b_d01);

  // Rasterize triangles based on the type

  if (v2.y == v3.y)
  {
    fillBottomFlatTriangle(v1, v2, v3,
                           v1.x, v1.y,
                           v2.x, v2.y,
                           v3.x, v3.y,
                           pixel_ptr, texturePitch, drawDepthBuffer,
                           color,
                           b_v0, b_v1, b_d00, b_d01, b_d11, b_invDenom);
  }
  else if (v1.y == v2.y)
  {
    fillTopFlatTriangle(v1, v2, v3,
                        v1.x, v1.y,
                        v2.x, v2.y,
                        v3.x, v3.y,
                        pixel_ptr, texturePitch, drawDepthBuffer,
                        color,
                        b_v0, b_v1, b_d00, b_d01, b_d11, b_invDenom);
  }
  else
  {
    float newX = v1.x + ((v2.y - v1.y) / (v3.y - v1.y)) * (v3.x - v1.x);

    fillBottomFlatTriangle(v1, v2, v3,
                           v1.x, v1.y,
                           v2.x, v2.y,
                           newX, v2.y, // New x and y
                           pixel_ptr, texturePitch, drawDepthBuffer,
                           color,
                           b_v0, b_v1, b_d00, b_d01, b_d11, b_invDenom);

    fillTopFlatTriangle(v1, v2, v3,
                        v2.x, v2.y,
                        newX, v2.y, // New x and y
                        v3.x, v3.y,
                        pixel_ptr, texturePitch, drawDepthBuffer,
                        color,
                        b_v0, b_v1, b_d00, b_d01, b_d11, b_invDenom);
  }
}