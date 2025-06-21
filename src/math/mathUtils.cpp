#include <cmath>

void sumArrays(float in[][3], int rows, float *val, float mult)
{
  for (int i = 0; i < rows; i++)
  {
    in[i][0] += (val[0] * mult);
    in[i][1] += (val[1] * mult);
    in[i][2] += (val[2] * mult);
  }
}

void matMult(float *in, float mat[3][3])
{
  float result[3];

  result[0] = in[0] * mat[0][0] + in[1] * mat[1][0] + in[2] * mat[2][0];
  result[1] = in[0] * mat[0][1] + in[1] * mat[1][1] + in[2] * mat[2][1];
  result[2] = in[0] * mat[0][2] + in[1] * mat[1][2] + in[2] * mat[2][2];

  in[0] = result[0];
  in[1] = result[1];
  in[2] = result[2];
}

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