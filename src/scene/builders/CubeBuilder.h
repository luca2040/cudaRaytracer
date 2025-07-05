#pragma once

#include "../SceneClasses.h"
#include "../../math/Operations.h"

SceneObject generateCube(float3_L center, float sideLenght,
                         float3_L defaultRot, TransformFunction trFunc, bool hasTransformFunction,
                         int face1Col,
                         int face2Col,
                         int face3Col,
                         int face4Col,
                         int face5Col,
                         int face6Col)
{
  std::vector<float3_L> cubePoints;

  float sideLenghtHalf = sideLenght / 2.0f;

  for (float frontRear = -1.0f; frontRear <= 1.0f; frontRear += 2.0f)
    for (float upDown = -1.0f; upDown <= 1.0f; upDown += 2.0f)
      for (float leftRight = -1.0f; leftRight <= 1.0f; leftRight += 2.0f)
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

  if (hasTransformFunction)
    return SceneObject(center, trFunc, cubePoints, cubeTriangles);
  else
    return SceneObject(center, defaultRot, cubePoints, cubeTriangles);
}

SceneObject generateCube(float3_L center, float sideLenght,
                         float3_L defaultRot,
                         int face1Col,
                         int face2Col,
                         int face3Col,
                         int face4Col,
                         int face5Col,
                         int face6Col)
{
  return generateCube(center, sideLenght,
                      defaultRot, nullptr, false,
                      face1Col, face2Col, face3Col, face4Col, face5Col, face6Col);
}

SceneObject generateCube(float3_L center, float sideLenght,
                         TransformFunction trFunc,
                         int face1Col,
                         int face2Col,
                         int face3Col,
                         int face4Col,
                         int face5Col,
                         int face6Col)
{
  return generateCube(center, sideLenght,
                      {}, trFunc, true,
                      face1Col, face2Col, face3Col, face4Col, face5Col, face6Col);
}

SceneObject generateCube(float3_L center, float sideLenght,
                         float3_L defaultRot,
                         int color)
{
  return generateCube(center, sideLenght,
                      defaultRot, nullptr, false,
                      color, color, color, color, color, color);
}

SceneObject generateCube(float3_L center, float sideLenght,
                         TransformFunction trFunc,
                         int color)
{
  return generateCube(center, sideLenght,
                      {}, trFunc, true,
                      color, color, color, color, color, color);
}