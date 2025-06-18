#include <algorithm>
#include <SDL2/SDL.h>

#include "../math/mathUtils.h"

void sortVerticesByY(
    float *x1, float *y1, float *z1,
    float *x2, float *y2, float *z2,
    float *x3, float *y3, float *z3)
{
  if (*y1 > *y2)
  {
    std::swap(*x1, *x2);
    std::swap(*y1, *y2);
    std::swap(*z1, *z2);
  }
  if (*y2 > *y3)
  {
    std::swap(*x2, *x3);
    std::swap(*y2, *y3);
    std::swap(*z2, *z3);
  }
  if (*y1 > *y2)
  {
    std::swap(*x1, *x2);
    std::swap(*y1, *y2);
    std::swap(*z1, *z2);
  }
}

void drawHorizontalLine(int xStart, int xEnd, int lineY,
                        float bigger_x1, float bigger_y1, float z1,
                        float bigger_x2, float bigger_y2, float z2,
                        float bigger_x3, float bigger_y3, float z3,
                        Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
                        int color,
                        int WIDTH, int HEIGHT)
{
  if (lineY > HEIGHT || lineY < 0)
    return;
  if (xStart > xEnd)
    std::swap(xStart, xEnd);

  for (int pointX = xStart; pointX <= xEnd; pointX++)
  {
    if (pointX > WIDTH || pointX < 0)
      continue;

    float pointZ = triangleInterpolate(bigger_x1, bigger_y1, bigger_x2, bigger_y2, bigger_x3, bigger_y3, pointX, lineY, z1, z2, z3);
    float &actualZ = drawDepthBuffer[lineY * WIDTH + pointX];

    if (actualZ < pointZ)
      continue;
    actualZ = pointZ;

    pixel_ptr[lineY * (texturePitch / 4) + pointX] = color;
  }
}

void fillBottomFlatTriangle(
    float bigger_x1, float bigger_y1, float z1,
    float bigger_x2, float bigger_y2, float z2,
    float bigger_x3, float bigger_y3, float z3,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3,
    Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
    int color,
    int WIDTH, int HEIGHT)
{
  float invslope1 = (x2 - x1) / (y2 - y1);
  float invslope2 = (x3 - x1) / (y3 - y1);

  float curx1 = x1;
  float curx2 = x1;

  for (int scanlineY = y1; scanlineY <= y2; scanlineY++)
  {
    // Another abomination again :)
    drawHorizontalLine(static_cast<int>(curx1), static_cast<int>(curx2), scanlineY,
                       bigger_x1, bigger_y1, z1,
                       bigger_x2, bigger_y2, z2,
                       bigger_x3, bigger_y3, z3,
                       pixel_ptr, texturePitch, drawDepthBuffer,
                       color,
                       WIDTH, HEIGHT);

    curx1 += invslope1;
    curx2 += invslope2;
  }
}

void fillTopFlatTriangle(
    float bigger_x1, float bigger_y1, float z1,
    float bigger_x2, float bigger_y2, float z2,
    float bigger_x3, float bigger_y3, float z3,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3,
    Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
    int color,
    int WIDTH, int HEIGHT)
{
  float invslope1 = (x3 - x1) / (y3 - y1);
  float invslope2 = (x3 - x2) / (y3 - y2);

  float curx1 = x3;
  float curx2 = x3;

  for (int scanlineY = y3; scanlineY > y1; scanlineY--)
  {
    // Another abomination again :)
    drawHorizontalLine(static_cast<int>(curx1), static_cast<int>(curx2), scanlineY,
                       bigger_x1, bigger_y1, z1,
                       bigger_x2, bigger_y2, z2,
                       bigger_x3, bigger_y3, z3,
                       pixel_ptr, texturePitch, drawDepthBuffer,
                       color,
                       WIDTH, HEIGHT);

    curx1 -= invslope1;
    curx2 -= invslope2;
  }
}

// Yes this is a part of that abomination idea in DrawLoop
void depthFillTriangle(
    float x1, float y1, float z1,
    float x2, float y2, float z2,
    float x3, float y3, float z3,
    Uint32 *pixel_ptr, int texturePitch, float *drawDepthBuffer,
    int color,
    int WIDTH, int HEIGHT)
{
  sortVerticesByY(&x1, &y1, &z1, &x2, &y2, &z2, &x3, &y3, &z3);

  if (y2 == y3)
  {
    fillBottomFlatTriangle(x1, y1, z1,
                           x2, y2, z2,
                           x3, y3, z3,
                           x1, y1,
                           x2, y2,
                           x3, y3,
                           pixel_ptr, texturePitch, drawDepthBuffer,
                           color,
                           WIDTH, HEIGHT);
  }
  else if (y1 == y2)
  {
    fillTopFlatTriangle(x1, y1, z1,
                        x2, y2, z2,
                        x3, y3, z3,
                        x1, y1,
                        x2, y2,
                        x3, y3,
                        pixel_ptr, texturePitch, drawDepthBuffer,
                        color,
                        WIDTH, HEIGHT);
  }
  else
  {
    int newX = static_cast<int>(x1 + (static_cast<float>(y2 - y1) / static_cast<float>(y3 - y1)) * (x3 - x1));
    int newY = y2;

    fillBottomFlatTriangle(x1, y1, z1,
                           x2, y2, z2,
                           x3, y3, z3,
                           x1, y1,
                           x2, y2,
                           newX, newY,
                           pixel_ptr, texturePitch, drawDepthBuffer,
                           color,
                           WIDTH, HEIGHT);

    fillTopFlatTriangle(x1, y1, z1,
                        x2, y2, z2,
                        x3, y3, z3,
                        x2, y2,
                        newX, newY,
                        x3, y3,
                        pixel_ptr, texturePitch, drawDepthBuffer,
                        color,
                        WIDTH, HEIGHT);
  }
}