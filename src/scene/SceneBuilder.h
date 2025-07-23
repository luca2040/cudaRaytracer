#pragma once

#include <vector>

#include "SceneClasses.h"
#include "structs/transformIndexPair.h"
#include "structs/SceneObject.h"
#include "structs/Scene.h"
#include "../math/Operations.h"

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
    bool isBBnew = true;

    for (auto currentVertex : objPoints)
    {
      if (isBBnew)
      {
        objBB.l = objBB.h = currentVertex;
        isBBnew = false;
        continue;
      }

      growBBtoInclude(objBB, currentVertex);
    }

    SceneObject newSceneObj = {currentTriangleIndex, afterListTriangleIndex, objBB};
    sceneObjects.push_back(newSceneObj);
    transformIndexPair toAddTransformPair = {currentPointIndex, afterListPointIndex, objTransform};
    groupIndexRanges.push_back(toAddTransformPair);

    return groupIndexRanges.size() - 1;
  }

  void compile()
  {
    computeNormals();

    scene.pointsCount = scenePoints.size();
    scene.points = new float3_L[scene.pointsCount];
    std::copy(scenePoints.begin(), scenePoints.end(), scene.points);

    scene.triangleNum = sceneTriangles.size();
    scene.triangles = new triangleidx[scene.triangleNum];
    std::copy(sceneTriangles.begin(), sceneTriangles.end(), scene.triangles);

    scene.trIndexPairCount = groupIndexRanges.size();
    scene.trIndexPairs = new transformIndexPair[scene.trIndexPairCount];
    std::copy(groupIndexRanges.begin(), groupIndexRanges.end(), scene.trIndexPairs);

    scene.sceneobjectsNum = sceneObjects.size();
    scene.sceneobjects = new SceneObject[scene.sceneobjectsNum];
    std::copy(sceneObjects.begin(), sceneObjects.end(), scene.sceneobjects);

    scene.dyntrianglesNum = dynamicTriangles.size();
    scene.dyntriangles = new size_t[scene.dyntrianglesNum];
    std::copy(dynamicTriangles.begin(), dynamicTriangles.end(), scene.dyntriangles);
  }
};