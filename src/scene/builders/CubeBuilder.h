#pragma once

#include "../SceneClasses.h"
#include "../../math/Operations.h"

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                         float3_L defaultRot, TransformFunction trFunc, bool hasTransformFunction,
                         int face1Col, float reflectiveness1,
                         int face2Col, float reflectiveness2,
                         int face3Col, float reflectiveness3,
                         int face4Col, float reflectiveness4,
                         int face5Col, float reflectiveness5,
                         int face6Col, float reflectiveness6)
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
      {0, 1, 3, face1Col, reflectiveness1},
      {0, 3, 2, face1Col, reflectiveness1},

      {1, 5, 3, face2Col, reflectiveness2},
      {5, 7, 3, face2Col, reflectiveness2},

      {0, 4, 2, face3Col, reflectiveness3},
      {4, 6, 2, face3Col, reflectiveness3},

      {4, 5, 7, face4Col, reflectiveness4},
      {4, 7, 6, face4Col, reflectiveness4},

      {0, 1, 5, face5Col, reflectiveness5},
      {0, 5, 4, face5Col, reflectiveness5},

      {2, 3, 7, face6Col, reflectiveness6},
      {2, 7, 6, face6Col, reflectiveness6},
  };

  if (hasTransformFunction)
    return SceneObjectPassthrough(center, trFunc, cubePoints, cubeTriangles);
  else
    return SceneObjectPassthrough(center, defaultRot, cubePoints, cubeTriangles);
}

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                         float3_L defaultRot,
                         int face1Col,
                         int face2Col,
                         int face3Col,
                         int face4Col,
                         int face5Col,
                         int face6Col,
                         float reflectiveness)
{
  return generateCube(center, sideLenght,
                      defaultRot, nullptr, false,
                      face1Col, reflectiveness,
                      face2Col, reflectiveness,
                      face3Col, reflectiveness,
                      face4Col, reflectiveness,
                      face5Col, reflectiveness,
                      face6Col, reflectiveness);
}

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                         float3_L defaultRot,
                         int face1Col, float reflectiveness1,
                         int face2Col, float reflectiveness2,
                         int face3Col, float reflectiveness3,
                         int face4Col, float reflectiveness4,
                         int face5Col, float reflectiveness5,
                         int face6Col, float reflectiveness6)
{
  return generateCube(center, sideLenght,
                      defaultRot, nullptr, false,
                      face1Col, reflectiveness1,
                      face2Col, reflectiveness2,
                      face3Col, reflectiveness3,
                      face4Col, reflectiveness4,
                      face5Col, reflectiveness5,
                      face6Col, reflectiveness6);
}

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                         TransformFunction trFunc,
                         int face1Col,
                         int face2Col,
                         int face3Col,
                         int face4Col,
                         int face5Col,
                         int face6Col,
                         float reflectiveness)
{
  return generateCube(center, sideLenght,
                      {}, trFunc, true,
                      face1Col, reflectiveness,
                      face2Col, reflectiveness,
                      face3Col, reflectiveness,
                      face4Col, reflectiveness,
                      face5Col, reflectiveness,
                      face6Col, reflectiveness);
}

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                         TransformFunction trFunc,
                         int face1Col, float reflectiveness1,
                         int face2Col, float reflectiveness2,
                         int face3Col, float reflectiveness3,
                         int face4Col, float reflectiveness4,
                         int face5Col, float reflectiveness5,
                         int face6Col, float reflectiveness6)
{
  return generateCube(center, sideLenght,
                      {}, trFunc, true,
                      face1Col, reflectiveness1,
                      face2Col, reflectiveness2,
                      face3Col, reflectiveness3,
                      face4Col, reflectiveness4,
                      face5Col, reflectiveness5,
                      face6Col, reflectiveness6);
}

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                         float3_L defaultRot,
                         int color, float reflectiveness)
{
  return generateCube(center, sideLenght,
                      defaultRot, nullptr, false,
                      color, reflectiveness,
                      color, reflectiveness,
                      color, reflectiveness,
                      color, reflectiveness,
                      color, reflectiveness,
                      color, reflectiveness);
}

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                         TransformFunction trFunc,
                         int color, float reflectiveness)
{
  return generateCube(center, sideLenght,
                      {}, trFunc, true,
                      color, reflectiveness,
                      color, reflectiveness,
                      color, reflectiveness,
                      color, reflectiveness,
                      color, reflectiveness,
                      color, reflectiveness);
}