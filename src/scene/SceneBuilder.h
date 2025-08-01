#pragma once

#include <vector>

#include "SceneClasses.h"
#include "structs/transformIndexPair.h"
#include "structs/SceneObject.h"
#include "structs/Scene.h"
#include "MaterialHandler.h"
#include "../math/Operations.h"

class SceneBuilder
{
private:
public:
  std::vector<float3_L> scenePoints;
  std::vector<size_t> pointToObjTable;
  std::vector<triangleidx> sceneTriangles;
  std::vector<size_t> dynamicTriangles;
  std::vector<SceneObject> sceneObjects;

  std::vector<transformIndexPair> groupIndexRanges; // When adding some points, save here the corresponding index start and end; the index of the pair is returned.

  MaterialHandler materials;

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

    size_t currentSceneObjIdx = sceneObjects.size();

    for (auto modelPoint : objPoints)
    {
      scenePoints.push_back(modelPoint);
      pointToObjTable.push_back(currentSceneObjIdx);
    }

    AABB objBB = {};
    for (auto currentVertex : objPoints)
    {
      growBBtoInclude(objBB, currentVertex);
    }

    SceneObject newSceneObj = {currentTriangleIndex, objToAdd.triangles.size(),
                               currentPointIndex, objToAdd.points.size(),
                               objBB};
    sceneObjects.push_back(newSceneObj);

    transformIndexPair toAddTransformPair = {objTransform, currentSceneObjIdx};
    groupIndexRanges.push_back(toAddTransformPair);

    return groupIndexRanges.size() - 1;
  }

  void compile()
  {
    scene->pointsCount = scenePoints.size();
    scene->points = new float3_L[scene->pointsCount];
    std::copy(scenePoints.begin(), scenePoints.end(), scene->points);

    scene->pointToObjIdxTable = new size_t[scene->pointsCount];
    std::copy(pointToObjTable.begin(), pointToObjTable.end(), scene->pointToObjIdxTable);

    scene->triangleNum = sceneTriangles.size();
    scene->triangles = new triangleidx[scene->triangleNum];
    std::copy(sceneTriangles.begin(), sceneTriangles.end(), scene->triangles);

    scene->trIndexPairCount = groupIndexRanges.size();
    scene->trIndexPairs = new transformIndexPair[scene->trIndexPairCount];
    std::copy(groupIndexRanges.begin(), groupIndexRanges.end(), scene->trIndexPairs);

    scene->sceneobjectsNum = sceneObjects.size();
    scene->sceneobjects = new SceneObject[scene->sceneobjectsNum];
    std::copy(sceneObjects.begin(), sceneObjects.end(), scene->sceneobjects);

    scene->materialsNum = materials.materials.size();
    scene->materials = new Material[scene->materialsNum];
    std::copy(materials.materials.begin(), materials.materials.end(), scene->materials);
  }
};