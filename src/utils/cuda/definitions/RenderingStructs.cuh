#pragma once

#include "../../../math/Definitions.h"
#include "../../../math/cuda/CudaMath.cuh"

struct RayData
{
  bool hasHit = false; // Gets true when the ray hits at least a triangle
  float3_L rayLight = make_float3_L(0.0f);
  float3_L color = make_float3_L(1.0f);
  float lastDiffuse = 1.0f;

  __host__ __device__ __forceinline__ RayData() = default;
};
