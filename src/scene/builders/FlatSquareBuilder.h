#pragma once

#include "../SceneClasses.h"
#include "../../math/Operations.h"

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

  if (firstCorner.x == secondCorner.x) // YZ plane
  {
    float2_L firstCornerPlane = {firstCorner.y, firstCorner.z};
    float2_L secondCornerPlane = {secondCorner.y, secondCorner.z};
    float equalXval = firstCorner.x;

    float2_L distance = secondCornerPlane - firstCornerPlane;
    float2_L dt = distance / static_cast<float>(sideDivisions);

    for (int divY = 0; divY < sideDivisions; divY++)
    {
      for (int divZ = 0; divZ < sideDivisions; divZ++)
      {
        float2_L currentPos = firstCornerPlane + float2_L(divY, divZ) * dt;
        float2_L nextPos = firstCornerPlane + (float2_L(divY, divZ) + 1) * dt;

        bool useFirstColor = (divY % 2 == 0) ^ (divZ % 2 == 1);
        int actualColor = useFirstColor ? col1 : col2;

        float3_L startPos = {equalXval, currentPos.x, currentPos.y};
        float3_L endPos = {equalXval, nextPos.x, nextPos.y};

        float3_L mixedVertex1 = {equalXval, currentPos.x, nextPos.y};
        float3_L mixedVertex2 = {equalXval, nextPos.x, currentPos.y};

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
  }
  else if (firstCorner.y == secondCorner.y) // XZ plane
  {
    float2_L firstCornerPlane = {firstCorner.x, firstCorner.z};
    float2_L secondCornerPlane = {secondCorner.x, secondCorner.z};
    float equalYval = firstCorner.y;

    float2_L distance = secondCornerPlane - firstCornerPlane;
    float2_L dt = distance / static_cast<float>(sideDivisions);

    for (int divX = 0; divX < sideDivisions; divX++)
    {
      for (int divZ = 0; divZ < sideDivisions; divZ++)
      {
        float2_L currentPos = firstCornerPlane + float2_L(divX, divZ) * dt;
        float2_L nextPos = firstCornerPlane + (float2_L(divX, divZ) + 1) * dt;

        bool useFirstColor = (divX % 2 == 0) ^ (divZ % 2 == 1);
        int actualColor = useFirstColor ? col1 : col2;

        float3_L startPos = {currentPos.x, equalYval, currentPos.y};
        float3_L endPos = {nextPos.x, equalYval, nextPos.y};

        float3_L mixedVertex1 = {currentPos.x, equalYval, nextPos.y};
        float3_L mixedVertex2 = {nextPos.x, equalYval, currentPos.y};

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
  }
  else // XY plane
  {
    float2_L firstCornerPlane = {firstCorner.x, firstCorner.y};
    float2_L secondCornerPlane = {secondCorner.x, secondCorner.y};
    float equalZval = firstCorner.z;

    float2_L distance = secondCornerPlane - firstCornerPlane;
    float2_L dt = distance / static_cast<float>(sideDivisions);

    for (int divX = 0; divX < sideDivisions; divX++)
    {
      for (int divY = 0; divY < sideDivisions; divY++)
      {
        float2_L currentPos = firstCornerPlane + float2_L(divX, divY) * dt;
        float2_L nextPos = firstCornerPlane + (float2_L(divX, divY) + 1) * dt;

        bool useFirstColor = (divX % 2 == 0) ^ (divY % 2 == 1);
        int actualColor = useFirstColor ? col1 : col2;

        float3_L startPos = {currentPos.x, currentPos.y, equalZval};
        float3_L endPos = {nextPos.x, nextPos.y, equalZval};

        float3_L mixedVertex1 = {currentPos.x, nextPos.y, equalZval};
        float3_L mixedVertex2 = {nextPos.x, currentPos.y, equalZval};

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
  }

  return SceneObject(center, defaultRot, facePoints, faceTriangles);
}