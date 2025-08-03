#pragma once

#include "../Definitions.h"

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