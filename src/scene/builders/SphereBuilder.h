#pragma once

#include "../SceneClasses.h"
#include "../../math/Operations.h"

SceneObjectPassthrough generateSphere(float3_L center, float radius,
                                      float3_L defaultRot, TransformFunction trFunc, bool hasTransformFunction,
                                      size_t material)
{
  Sphere actualSphere = Sphere(center, radius);
  actualSphere.materialIdx = material;
  if (hasTransformFunction)
  {
    return SceneObjectPassthrough(center, trFunc, actualSphere);
  }
  else
  {
    return SceneObjectPassthrough(center, defaultRot, actualSphere);
  }
}

SceneObjectPassthrough generateSphere(float3_L center, float radius,
                                      float3_L defaultRot, size_t material)
{
  return generateSphere(center, radius, defaultRot, nullptr, false, material);
}

SceneObjectPassthrough generateSphere(float3_L center, float radius,
                                      TransformFunction trFunc, size_t material)
{
  return generateSphere(center, radius, {}, trFunc, true, material);
}