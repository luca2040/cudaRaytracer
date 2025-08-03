#pragma once

#include "../../../math/Definitions.h"
#include "../../../math/cuda/CudaMath.cuh"

struct RayData
{
  float3_L rayLight = make_float3_L(0.0f, 0.0f, 0.0f);
  float3_L color = make_float3_L(1.0f, 1.0f, 1.0f);
  float reflReduction = 1.0f;

  __host__ __device__ __forceinline__ RayData() = default;
};
