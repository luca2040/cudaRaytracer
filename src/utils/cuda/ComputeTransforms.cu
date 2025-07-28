#include "ComputeTransforms.cuh"

#include "../DrawValues.h"
#include "../../scene/structs/Scene.h"
#include "kernels/transform/TransformKernel.cuh"

#include "../../third_party/tracy/tracy/Tracy.hpp"
#include "../../third_party/tracy/tracy/TracyC.h"

#include <iostream>

void computeDeviceTransforms()
{
  ZoneScopedN("computeDeviceTransforms function");
  TracyCZoneN(transformsMemcpy, "Transforms cudamemcpy", true);

  cudaMemcpy(scene->d_transformMatrices, scene->transformMatrices, scene->matricesSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scene, scene, sceneStructSize, cudaMemcpyHostToDevice);

  TracyCZoneEnd(transformsMemcpy);

  size_t threadsPerBlock = VERT_THREADS_PER_BLOCK;

  TracyCZoneN(resetbbsKernelRun, "Reset AABBs kernel", true);

  // Launch a thread per model
  size_t numResetBlocks = (scene->sceneobjectsNum + threadsPerBlock - 1) / threadsPerBlock;
  resetAABBsKernel<<<numResetBlocks, threadsPerBlock>>>(d_scene);

  TracyCZoneEnd(resetbbsKernelRun);

  TracyCZoneN(transformKernelRun, "Transforms kernel", true);

  // Launch a thread per vert
  size_t numTransformBlocks = (scene->pointsCount + threadsPerBlock - 1) / threadsPerBlock;
  transformKernel<<<numTransformBlocks, threadsPerBlock>>>(d_scene);

  TracyCZoneEnd(transformKernelRun);

  TracyCZoneN(normalsKernelRun, "Normals kernel", true);

  // Launch a thread per triangle
  size_t numNormalBlocks = (scene->triangleNum + threadsPerBlock - 1) / threadsPerBlock;
  normalComputeKernel<<<numNormalBlocks, threadsPerBlock>>>(d_scene);

  TracyCZoneEnd(normalsKernelRun);
}