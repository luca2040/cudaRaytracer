#include <glm/glm.hpp>

inline float triangleInterpolate(glm::vec3 t_v1, glm::vec3 t_v2, glm::vec3 t_v3,
                                 float px, float py,
                                 glm::vec2 &v0, glm::vec2 &v1, float &d00, float &d01, float &d11, float &invDenom)
{
  glm::vec2 v2 = glm::vec2(px, py) - glm::vec2(t_v1);
  float d20 = v2.x * v0.x + v2.y * v0.y;
  float d21 = v2.x * v1.x + v2.y * v1.y;
  float v = (d11 * d20 - d01 * d21) * invDenom;
  float w = (d00 * d21 - d01 * d20) * invDenom;
  float u = 1.0f - v - w;

  return t_v1.z * u + t_v2.z * v + t_v3.z * w;
}