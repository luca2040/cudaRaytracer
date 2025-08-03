#pragma once

#include "CudaMath.cuh"

__device__ __forceinline__ uint nextRandom(uint &state)
{
  state = state * 747796405 + 2891336453;
  uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
  result = (result >> 22) ^ result;
  return result;
}

// Returns a random float between 0.0f and 1.0f
__device__ __forceinline__ float randomValue(uint &state)
{
  return nextRandom(state) / 4294967295.0; // 2^32 - 1
}

// Returns a random float between -1.0f and 1.0f
__device__ __forceinline__ float balancedRandomValue(uint &state)
{
  return randomValue(state) * 2.0f - 1.0f;
}

// Returns a normalized random vector
__device__ __forceinline__ float3_L randomVector(uint &state)
{
  return normalize3_cuda(make_float3_L(
      balancedRandomValue(state),
      balancedRandomValue(state),
      balancedRandomValue(state)));
}