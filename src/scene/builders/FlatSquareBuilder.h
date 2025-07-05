#pragma once

#include "../SceneClasses.h"
#include "../../math/Operations.h"

enum PlaneType
{
  YZ,
  XZ,
  XY
};

SceneObject generateFlatSquare(float3_L firstCorner, float3_L secondCorner,
                               float3_L defaultRot,
                               int sideDivisions,
                               int col1,
                               int col2)
{
  float3_L center = (firstCorner + secondCorner) * 0.5f;

  std::vector<float3_L> facePoints;
  std::vector<triangleidx> faceTriangles;

  size_t currentVertIdx = 0;

  PlaneType plane;

  if (firstCorner.x == secondCorner.x)
    plane = YZ;
  else if (firstCorner.y == secondCorner.y)
    plane = XZ;
  else
    plane = XY;

  float2_L firstCornerPlane;
  float2_L secondCornerPlane;
  float equalVal;

  switch (plane)
  {
  case YZ:
    firstCornerPlane = {firstCorner.y, firstCorner.z};
    secondCornerPlane = {secondCorner.y, secondCorner.z};
    equalVal = firstCorner.x;
    break;
  case XZ:
    firstCornerPlane = {firstCorner.x, firstCorner.z};
    secondCornerPlane = {secondCorner.x, secondCorner.z};
    equalVal = firstCorner.y;
    break;
  case XY:
    firstCornerPlane = {firstCorner.x, firstCorner.y};
    secondCornerPlane = {secondCorner.x, secondCorner.y};
    equalVal = firstCorner.z;
    break;
  }

  float2_L distance = secondCornerPlane - firstCornerPlane;
  float2_L dt = distance / static_cast<float>(sideDivisions);

  for (int divA = 0; divA < sideDivisions; divA++)
  {
    for (int divB = 0; divB < sideDivisions; divB++)
    {
      float2_L currentPos = firstCornerPlane + float2_L(divA, divB) * dt;
      float2_L nextPos = firstCornerPlane + (float2_L(divA, divB) + 1) * dt;

      bool useFirstColor = (divA % 2 == 0) ^ (divB % 2 == 1);
      int actualColor = useFirstColor ? col1 : col2;

      float3_L startPos;
      float3_L endPos;
      float3_L mixedVertex1;
      float3_L mixedVertex2;

      switch (plane)
      {
      case YZ:
        startPos = {equalVal, currentPos.x, currentPos.y};
        endPos = {equalVal, nextPos.x, nextPos.y};
        mixedVertex1 = {equalVal, currentPos.x, nextPos.y};
        mixedVertex2 = {equalVal, nextPos.x, currentPos.y};
        break;
      case XZ:
        startPos = {currentPos.x, equalVal, currentPos.y};
        endPos = {nextPos.x, equalVal, nextPos.y};
        mixedVertex1 = {currentPos.x, equalVal, nextPos.y};
        mixedVertex2 = {nextPos.x, equalVal, currentPos.y};
        break;
      case XY:
        startPos = {currentPos.x, currentPos.y, equalVal};
        endPos = {nextPos.x, nextPos.y, equalVal};
        mixedVertex1 = {currentPos.x, nextPos.y, equalVal};
        mixedVertex2 = {nextPos.x, currentPos.y, equalVal};
        break;
      }

      facePoints.push_back(startPos);     // 0
      facePoints.push_back(endPos);       // 1
      facePoints.push_back(mixedVertex1); // 2
      facePoints.push_back(mixedVertex2); // 3

      faceTriangles.push_back({2 + currentVertIdx,
                               3 + currentVertIdx,
                               0 + currentVertIdx,
                               actualColor});
      faceTriangles.push_back({2 + currentVertIdx,
                               1 + currentVertIdx,
                               3 + currentVertIdx,
                               actualColor});

      currentVertIdx += 4;
    }
  }

  return SceneObject(center, defaultRot, facePoints, faceTriangles);
}

SceneObject generateFlatSquare(float3_L firstCorner, float3_L secondCorner,
                               int sideDivisions,
                               int col1,
                               int col2)
{
  return generateFlatSquare(firstCorner, secondCorner,
                            {0, 0, 0},
                            sideDivisions,
                            col1, col2);
}

SceneObject generateFlatSquare(float3_L firstCorner, float3_L secondCorner,
                               int sideDivisions,
                               int col)
{
  return generateFlatSquare(firstCorner, secondCorner,
                            {0, 0, 0},
                            sideDivisions,
                            col, col);
}

SceneObject generateFlatSquare(float3_L firstCorner, float3_L secondCorner,
                               int col)
{
  return generateFlatSquare(firstCorner, secondCorner,
                            {0, 0, 0}, 1,
                            col, col);
}