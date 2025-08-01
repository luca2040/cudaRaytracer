#pragma once

#include <xmmintrin.h>
#include "Definitions.h"

template <typename T>
size_t getItemSize(size_t lenght)
{
  return sizeof(T) * lenght;
}

inline float clamp(float val, float min, float max) noexcept
{
  return val < min ? min : (val > max ? max : val);
}

inline int2_L operator+(const int2_L &a, const int2_L &b) noexcept
{
  return {a.x + b.x, a.y + b.y};
}

inline int2_L operator-(const int2_L &a, const int2_L &b) noexcept
{
  return {a.x - b.x, a.y - b.y};
}

inline float2_L operator+(const float2_L &a, const float2_L &b) noexcept
{
  return {a.x + b.x, a.y + b.y};
}

inline float2_L operator+(const float2_L &a, const float &b) noexcept
{
  return {a.x + b, a.y + b};
}

inline float2_L operator-(const float2_L &a, const float2_L &b) noexcept
{
  return {a.x - b.x, a.y - b.y};
}

inline float2_L operator*(const float2_L &a, const float &b) noexcept
{
  return {a.x * b, a.y * b};
}

inline float2_L operator*(const float2_L &a, const float2_L &b) noexcept
{
  return {a.x * b.x, a.y * b.y};
}

inline float2_L operator/(const float2_L &a, const float &b) noexcept
{
  return {a.x / b, a.y / b};
}

inline float dot2(const float2_L &a, const float2_L &b) noexcept
{
  return a.x * b.x + a.y * b.y;
}

inline float dot3(const float3_L &a, const float3_L &b) noexcept
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3_L cross3(const float3_L &a, const float3_L &b) noexcept
{
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x};
}

inline float3_L operator*(const float3_L &a, const float &b) noexcept
{
  return {a.x * b, a.y * b, a.z * b};
}

inline float3_L operator+(const float3_L &a, const float3_L &b) noexcept
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline float3_L operator-(const float &a, const float3_L &b) noexcept
{
  return {a - b.x, a - b.y, a - b.z};
}

inline float3_L operator-(const float3_L &a, const float3_L &b) noexcept
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline float3_L normalize(float3_L v) noexcept
{
  __m128 vec = _mm_set_ps(0.0f, v.z, v.y, v.x); // Set _, z, y, x

  // Dot product: x*x + y*y + z*z
  __m128 mul = _mm_mul_ps(vec, vec);
  __m128 shuf = _mm_movehl_ps(mul, mul);
  __m128 sum = _mm_add_ps(mul, shuf);
  shuf = _mm_shuffle_ps(sum, sum, 1);
  sum = _mm_add_ss(sum, shuf); // x*x + y*y + z*z

  __m128 inv_sqrt = _mm_rsqrt_ss(sum); // Inverse sqrt

  inv_sqrt = _mm_shuffle_ps(inv_sqrt, inv_sqrt, 0x00);
  __m128 normalized = _mm_mul_ps(vec, inv_sqrt); // Multiply by inverse sqrt

  // Get the result back in a float3
  float3_L result;
  _mm_store_ss(&result.x, normalized);
  _mm_store_ss(&result.y, _mm_shuffle_ps(normalized, normalized, _MM_SHUFFLE(1, 1, 1, 1)));
  _mm_store_ss(&result.z, _mm_shuffle_ps(normalized, normalized, _MM_SHUFFLE(2, 2, 2, 2)));
  return result;
}

inline const mat4x4 make4x4IdentityMat()
{
  return {{{1.f, 0.f, 0.f, 0.f},
           {0.f, 1.f, 0.f, 0.f},
           {0.f, 0.f, 1.f, 0.f},
           {0.f, 0.f, 0.f, 1.f}}};
}

inline mat4x4 translation4x4mat(const float3_L &translation) noexcept
{
  return {
      float4_L(1.f, 0.f, 0.f, translation.x),
      float4_L(0.f, 1.f, 0.f, translation.y),
      float4_L(0.f, 0.f, 1.f, translation.z),
      float4_L(0.f, 0.f, 0.f, 1.f)};
}