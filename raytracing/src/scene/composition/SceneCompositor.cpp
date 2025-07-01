#include "../../utils/DrawValues.h"
#include "../SceneBuilder.h"

#include "../builders/CubeBuilder.h"

void composeScene(float3_L *&pointarray, size_t &pointCount,
                  triangleidx *&triangles, size_t &triangleCount,
                  transformIndexPair *&indexpairs, size_t &indexPairCount,
                  DrawingLoopValues &loopValues)
{
  SceneBuilder builder;

  loopValues.simpleCubeIndex = builder.addObjectToScene(generateCube({0.0f, 0.0f, 3000.0f}, 2000.0f,
                                                                     0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF));

  builder.compile(pointarray, pointCount, triangles, triangleCount, indexpairs, indexPairCount);
}