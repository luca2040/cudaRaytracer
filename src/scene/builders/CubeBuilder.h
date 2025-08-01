#pragma once

#include "../SceneClasses.h"
#include "../../math/Operations.h"

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                                    float3_L defaultRot, TransformFunction trFunc, bool hasTransformFunction,
                                    size_t material1, size_t material2, size_t material3,
                                    size_t material4, size_t material5, size_t material6)
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
      {0, 1, 3, material1},
      {0, 3, 2, material1},

      {1, 5, 3, material2},
      {5, 7, 3, material2},

      {0, 4, 2, material3},
      {4, 6, 2, material3},

      {4, 5, 7, material4},
      {4, 7, 6, material4},

      {0, 1, 5, material5},
      {0, 5, 4, material5},

      {2, 3, 7, material6},
      {2, 7, 6, material6},
  };

  if (hasTransformFunction)
    return SceneObjectPassthrough(center, trFunc, cubePoints, cubeTriangles);
  else
    return SceneObjectPassthrough(center, defaultRot, cubePoints, cubeTriangles);
}

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                                    float3_L defaultRot,
                                    size_t material1, size_t material2, size_t material3,
                                    size_t material4, size_t material5, size_t material6)
{
  return generateCube(center, sideLenght,
                      defaultRot, nullptr, false,
                      material1, material2, material3,
                      material4, material5, material6);
}

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                                    TransformFunction trFunc,
                                    size_t material1, size_t material2, size_t material3,
                                    size_t material4, size_t material5, size_t material6)
{
  return generateCube(center, sideLenght,
                      {}, trFunc, true,
                      material1, material2, material3,
                      material4, material5, material6);
}

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                                    float3_L defaultRot,
                                    size_t material)
{
  return generateCube(center, sideLenght,
                      defaultRot, nullptr, false,
                      material, material, material,
                      material, material, material);
}

SceneObjectPassthrough generateCube(float3_L center, float sideLenght,
                                    TransformFunction trFunc,
                                    size_t material)
{
  return generateCube(center, sideLenght,
                      {}, trFunc, true,
                      material, material, material,
                      material, material, material);
}