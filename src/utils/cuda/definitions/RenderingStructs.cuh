#pragma once

#include "../../../math/Definitions.h"
#include "../../../math/cuda/CudaMath.cuh"

struct SceneMemoryPointers
{
  const float3_L *pointarray;
  const triangleidx *triangles;
  size_t triangleNum;

  __host__ __device__ __forceinline__ SceneMemoryPointers(const float3_L *pointarray, const triangleidx *triangles, size_t triangleNum)
      : pointarray(pointarray), triangles(triangles), triangleNum(triangleNum) {}
};

struct RayData
{
  float3_L color = make_float3_L(0.0f, 0.0f, 0.0f);
  float rayLight = 1.0f;

  __host__ __device__ __forceinline__ RayData() = default;
};
