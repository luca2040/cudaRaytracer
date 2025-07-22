#pragma once

#include <vector>

#include "SceneClasses.h"
#include "structs/SceneObject.h"
#include "../math/Operations.h"

struct transformIndexPair
{
  size_t startIdx;
  size_t endIdx;
  ObjTransform transform;

  inline transformIndexPair() = default;
  inline transformIndexPair(size_t start, size_t end, const ObjTransform &transform)
      : startIdx(start), endIdx(end), transform(transform) {}
};

class SceneBuilder
{
private:
  void computeNormals()
  {
    for (auto &triangle : sceneTriangles)
    {
      float3_L v1 = scenePoints[triangle.v1];
      float3_L v2 = scenePoints[triangle.v2];
      float3_L v3 = scenePoints[triangle.v3];

      float3_L side1 = v2 - v1;
      float3_L side2 = v3 - v1;

      triangle.normal = normalize(cross3(side1, side2));
    }
  }

public:
  std::vector<float3_L> scenePoints;
  std::vector<triangleidx> sceneTriangles;
  std::vector<size_t> dynamicTriangles;
  std::vector<SceneObject> sceneObjects;

  std::vector<transformIndexPair> groupIndexRanges; // When adding some points, save here the corresponding index start and end; the index of the pair is returned.

  size_t addObjectToScene(SceneObjectPassthrough objToAdd)
  {
    ObjTransform objTransform;

    if (objToAdd.hasTransformFunction)
    {
      objTransform = ObjTransform(objToAdd.rotationCenter, objToAdd.trFunc);
    }
    else
    {
      objTransform = ObjTransform(objToAdd.rotationCenter, objToAdd.defaultRot);
    }

    std::vector<float3_L> objPoints = objToAdd.points;
    size_t currentPointIndex = scenePoints.size();
    size_t currentTriangleIndex = sceneTriangles.size();

    for (auto oldTriangle : objToAdd.triangles)
    {
      triangleidx newTriangleIdx = oldTriangle;

      newTriangleIdx.v1 += currentPointIndex;
      newTriangleIdx.v2 += currentPointIndex;
      newTriangleIdx.v3 += currentPointIndex;

      sceneTriangles.push_back(newTriangleIdx); // Add triangles to the scene triangles vector

      if (objToAdd.hasTransformFunction)
      {
        size_t currentTriangleIdx = sceneTriangles.size() - 1;
        dynamicTriangles.push_back(currentTriangleIdx);
      }
    }

    scenePoints.insert(scenePoints.end(), objPoints.begin(), objPoints.end()); // Add obj's points to scene points

    size_t afterListPointIndex = scenePoints.size();
    size_t afterListTriangleIndex = sceneTriangles.size();

    AABB objBB = {};
    setBBtoNew(objBB);

    for (auto currentVertex : objPoints)
    {
      growBBtoInclude(objBB, currentVertex);
    }

    SceneObject newSceneObj = {currentTriangleIndex, afterListTriangleIndex, objBB};
    sceneObjects.push_back(newSceneObj);
    transformIndexPair toAddTransformPair = {currentPointIndex, afterListPointIndex, objTransform};
    groupIndexRanges.push_back(toAddTransformPair);

    return groupIndexRanges.size() - 1;
  }

  void compile(float3_L *&pointarray, size_t &pointCount,
               triangleidx *&triangles, size_t &triangleCount,
               transformIndexPair *&indexpairs, size_t &indexPairCount,
               SceneObject *&sceneobjects, size_t &sceneobjectsCount,
               size_t *&dyntriangles, size_t &dynTrianglesCount)
  {
    computeNormals();

    pointCount = scenePoints.size();
    pointarray = new float3_L[pointCount];
    std::copy(scenePoints.begin(), scenePoints.end(), pointarray);

    triangleCount = sceneTriangles.size();
    triangles = new triangleidx[triangleCount];
    std::copy(sceneTriangles.begin(), sceneTriangles.end(), triangles);

    indexPairCount = groupIndexRanges.size();
    indexpairs = new transformIndexPair[indexPairCount];
    std::copy(groupIndexRanges.begin(), groupIndexRanges.end(), indexpairs);

    sceneobjectsCount = sceneObjects.size();
    sceneobjects = new SceneObject[sceneobjectsCount];
    std::copy(sceneObjects.begin(), sceneObjects.end(), sceneobjects);

    dynTrianglesCount = dynamicTriangles.size();
    dyntriangles = new size_t[dynTrianglesCount];
    std::copy(dynamicTriangles.begin(), dynamicTriangles.end(), dyntriangles);
  }
};