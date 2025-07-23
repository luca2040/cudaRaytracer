#pragma once

#include "../../definitions/RenderingStructs.cuh"

// Generic hsv to rgb function
__device__ __forceinline__ float3_L hsv2rgb(float h, float s, float v)
{
  float c = v * s;
  float x = c * (1 - fabsf(fmodf(h * 6.0f, 2.0f) - 1));
  float m = v - c;
  float3_L rgb;

  // clang-format off
  rgb = (h < 1.0f / 6) ? make_float3_L(c, x, 0) :
        (h < 2.0f / 6) ? make_float3_L(x, c, 0) :
        (h < 3.0f / 6) ? make_float3_L(0, c, x) :
        (h < 4.0f / 6) ? make_float3_L(0, x, c) :
        (h < 5.0f / 6) ? make_float3_L(x, 0, c) :
                         make_float3_L(c, 0, x);
  // clang-format on

  rgb.x += m;
  rgb.y += m;
  rgb.z += m;
  return rgb;
}

// Map each index to an unique color (knowing the max index number) - used to debug
__device__ __forceinline__ float3_L indexToColor(uint objNum, uint maxObjNum)
{
  float hue = static_cast<float>(objNum) / static_cast<float>(maxObjNum);
  return hsv2rgb(hue, 1.0f, 1.0f);
}
