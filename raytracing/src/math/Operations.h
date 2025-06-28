#ifndef MATH_OPERATIONS
#define MATH_OPERATIONS

#include <xmmintrin.h>
#include <optional>

#include "Definitions.h"

inline float2_L operator+(const float2_L &a, const float2_L &b) noexcept
{
  return {a.x + b.x, a.y + b.y};
}

inline float2_L operator-(const float2_L &a, const float2_L &b) noexcept
{
  return {a.x - b.x, a.y - b.y};
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

inline float distance(const float3_L &a, const float3_L &b) noexcept
{
  __m128 va = _mm_set_ps(0.0f, a.z, a.y, a.x);
  __m128 vb = _mm_set_ps(0.0f, b.z, b.y, b.x);
  __m128 diff = _mm_sub_ps(va, vb);
  __m128 mul = _mm_mul_ps(diff, diff);
  __m128 shuf = _mm_movehl_ps(mul, mul);
  __m128 sum = _mm_add_ps(mul, shuf);
  shuf = _mm_shuffle_ps(sum, sum, 1);
  sum = _mm_add_ss(sum, shuf);
  __m128 sqrt = _mm_sqrt_ss(sum);
  return _mm_cvtss_f32(sqrt);
}

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
inline std::optional<float3_L> ray_intersects_triangle(const float3_L &ray_origin,
                                                     const float3_L &ray_vector,
                                                     const float3_L &tri_a,
                                                     const float3_L &tri_b,
                                                     const float3_L &tri_c) noexcept
{
  constexpr float epsilon = std::numeric_limits<float>::epsilon();

  float3_L edge1 = tri_b - tri_a;
  float3_L edge2 = tri_c - tri_a;
  float3_L ray_cross_e2 = cross3(ray_vector, edge2);
  float det = dot3(edge1, ray_cross_e2);

  if (det > -epsilon && det < epsilon)
    return std::nullopt; // This ray is parallel to this triangle.

  float inv_det = 1.0 / det;
  float3_L s = ray_origin - tri_a;
  float u = inv_det * dot3(s, ray_cross_e2);

  if ((u < 0 && abs(u) > epsilon) || (u > 1 && abs(u - 1) > epsilon))
    return std::nullopt;

  float3_L s_cross_e1 = cross3(s, edge1);
  float v = inv_det * dot3(ray_vector, s_cross_e1);

  if ((v < 0 && abs(v) > epsilon) || (u + v > 1 && abs(u + v - 1) > epsilon))
    return std::nullopt;

  // At this stage we can compute t to find out where the intersection point is on the line.
  float t = inv_det * dot3(edge2, s_cross_e1);

  if (t > epsilon) // ray intersection
  {
    return float3_L(ray_origin + ray_vector * t);
  }
  else // This means that there is a line intersection but not a ray intersection.
    return std::nullopt;
}

#endif