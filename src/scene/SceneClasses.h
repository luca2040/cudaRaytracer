#pragma once

#include <vector>
#include <cmath>

#include "../math/Definitions.h"

// Time, rotationAngles, relativePos
typedef void (*TransformFunction)(u_int32_t, float3_L &, float3_L &);

class SceneObjectPassthrough
{
public:
  float3_L rotationCenter;
  float3_L defaultRot;

  bool hasTransformFunction = false;
  TransformFunction trFunc;

  std::vector<float3_L> points;
  std::vector<triangleidx> triangles;

  SceneObjectPassthrough(float3_L rotationCenter_, float3_L defaultRot_,
              std::vector<float3_L> points_, std::vector<triangleidx> triangles_)
  {
    rotationCenter = rotationCenter_;
    defaultRot = defaultRot_;
    points = points_;
    triangles = triangles_;
  }
  SceneObjectPassthrough(float3_L rotationCenter_, TransformFunction trFunc_,
              std::vector<float3_L> points_, std::vector<triangleidx> triangles_)
  {
    rotationCenter = rotationCenter_;
    points = points_;
    triangles = triangles_;

    trFunc = trFunc_;
    hasTransformFunction = true;
  }
};

class ObjTransform
{
public:
  float3_L rotationCenter;
  float3_L rotationAngles = {0, 0, 0};
  float3_L relativePos = {0, 0, 0};

  bool hasTransformFunction = false;
  TransformFunction trFunc;

  ObjTransform() : rotationCenter{0, 0, 0}, rotationAngles{0, 0, 0} {}
  ObjTransform(float3_L rotationCenter_, float3_L rotationAngles_)
  {
    rotationCenter = rotationCenter_;
    rotationAngles = rotationAngles_;
  }
  ObjTransform(float3_L rotationCenter_, TransformFunction trFunc_)
  {
    rotationCenter = rotationCenter_;
    trFunc = trFunc_;
    hasTransformFunction = true;
  }

  inline mat3x3 getRotationMatrix()
  {
    float cx = std::cos(rotationAngles.x), sx = std::sin(rotationAngles.x);
    float cy = std::cos(rotationAngles.y), sy = std::sin(rotationAngles.y);
    float cz = std::cos(rotationAngles.z), sz = std::sin(rotationAngles.z);

    // Combined rotation matrix: xrot * yrot * zrot
    return {
        float3_L(
            cy * cz,
            -cy * sz,
            sy),
        float3_L(
            sx * sy * cz + cx * sz,
            -sx * sy * sz + cx * cz,
            -sx * cy),
        float3_L(
            -cx * sy * cz + sx * sz,
            cx * sy * sz + sx * cz,
            cx * cy)};
  }
};