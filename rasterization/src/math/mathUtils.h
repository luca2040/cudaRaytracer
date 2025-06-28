#include "names.h"

inline float triangleInterpolate(float3_L t_v1, float3_L t_v2, float3_L t_v3,
                                 float px, float py,
                                 float2_L &v0, float2_L &v1, float &d00, float &d01, float &d11, float &invDenom)
{
  float2_L v2 = float2_L(px, py) - t_v1;
  float d20 = dot(v2, v0);
  float d21 = dot(v2, v1);
  float v = (d11 * d20 - d01 * d21) * invDenom;
  float w = (d00 * d21 - d01 * d20) * invDenom;
  float u = 1.0f - v - w;

  return t_v1.z * u + t_v2.z * v + t_v3.z * w;
}