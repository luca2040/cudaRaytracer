#include <cmath>

inline float pointLineDistance(float ax, float ay, float bx, float by, float px, float py)
{
  float deltaX = bx - ax;
  float deltaY = by - ay;

  return abs(px * (by - ay) - py * (bx - ax) + bx * ay - by * ax) / sqrt(deltaY * deltaY + deltaX * deltaX);
}

float triangleInterpolate(float ax, float ay, float bx, float by, float cx, float cy, float px, float py, float aVal, float bVal, float cVal)
{
  float abcLineDistance;
  float pLineDistance;

  // First round: CB - A

  abcLineDistance = pointLineDistance(cx, cy, bx, by, ax, ay);
  pLineDistance = pointLineDistance(cx, cy, bx, by, px, py);

  float aTermPercent = pLineDistance / abcLineDistance;

  // Second round: CA - B

  abcLineDistance = pointLineDistance(cx, cy, ax, ay, bx, by);
  pLineDistance = pointLineDistance(cx, cy, ax, ay, px, py);

  float bTermPercent = pLineDistance / abcLineDistance;

  // Third round: AB - C

  // abcLineDistance = pointLineDistance(ax, ay, bx, by, cx, cy);
  // pLineDistance = pointLineDistance(ax, ay, bx, by, px, py);

  float cTermPercent = 1.0f - aTermPercent - bTermPercent;

  return aVal * aTermPercent + bVal * bTermPercent + cVal * cTermPercent;
}