#pragma once

#include "../../../math/Definitions.h"
#include "../../../scene/structs/SceneObject.h"
#include "../../../math/cuda/CudaMath.cuh"

struct SceneMemoryPointers
{
  const float3_L *pointarray;
  const triangleidx *triangles;
  const SceneObject *sceneobjects;
  size_t sceneobjectNum;

  __host__ __device__ __forceinline__ SceneMemoryPointers(const float3_L *pointarray, const triangleidx *triangles, const SceneObject *sceneobjects, size_t sceneobjectNum)
      : pointarray(pointarray), triangles(triangles), sceneobjects(sceneobjects), sceneobjectNum(sceneobjectNum) {}
};

struct RayData
{
  float3_L color = make_float3_L(0.0f, 0.0f, 0.0f);
  float rayLight = 1.0f;

  __host__ __device__ __forceinline__ RayData() = default;
};
