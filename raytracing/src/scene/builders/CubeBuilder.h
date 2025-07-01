#pragma once

#include "../SceneClasses.h"
#include "../../math/Operations.h"

SceneObject generateCube(float3_L center, float sideLenght,
                         unsigned int face1Col,
                         unsigned int face2Col,
                         unsigned int face3Col,
                         unsigned int face4Col,
                         unsigned int face5Col,
                         unsigned int face6Col)
{
  std::vector<float3_L> cubePoints;

  float sideLenghtHalf = sideLenght / 2.0f;

  for (float frontRear = -1.0f; frontRear == -1.0f; frontRear = 1.0f)
    for (float upDown = -1.0f; upDown == -1.0f; upDown = 1.0f)
      for (float leftRight = -1.0f; leftRight == -1.0f; leftRight = 1.0f)
      {
        cubePoints.push_back(center + float3_L(leftRight, upDown, frontRear) * sideLenghtHalf);
      }

  std::vector<triangleidx> cubeTriangles = {
      {0, 1, 3, face1Col},
      {0, 3, 2, face1Col},

      {1, 5, 3, face2Col},
      {5, 7, 3, face2Col},

      {0, 4, 2, face3Col},
      {4, 6, 2, face3Col},

      {4, 5, 7, face4Col},
      {4, 7, 6, face4Col},

      {0, 1, 5, face5Col},
      {0, 5, 4, face5Col},

      {2, 3, 7, face6Col},
      {2, 7, 6, face6Col},
  };

  return SceneObject(center, cubePoints, cubeTriangles);
}

SceneObject generateCube(float3_L center, float sideLenght,
                         unsigned int color)
{
  return generateCube(center, sideLenght,
                      color, color, color, color, color, color);
}