#include "ComputeTransforms.cuh"

#include "../DrawValues.h"
#include "../../scene/structs/Scene.h"
#include "kernels/transform/TransformKernel.cuh"

#include "../Profiling.h"

#include <iostream>

void computeDeviceTransforms()
{
  ZONESCOPEDNC("computeDeviceTransforms function", PROFILER_PINK);
  TRACYCZONENC(transformsMemcpy, "Transforms cudamemcpy", true, PROFILER_RED);

  cudaMemcpyAsync(d_scene, scene, sceneStructSize, cudaMemcpyHostToDevice);

  if (scene->transformSync)
    cudaDeviceSynchronize();

  TRACYCZONEEND(transformsMemcpy);

  size_t threadsPerBlock = VERT_THREADS_PER_BLOCK;

  TRACYCZONENC(resetbbsKernelRun, "Reset AABBs kernel", true, PROFILER_DARK_GREEN);

  // Launch a thread per model
  size_t numResetBlocks = (scene->sceneobjectsNum + threadsPerBlock - 1) / threadsPerBlock;
  resetAABBsKernel<<<numResetBlocks, threadsPerBlock>>>(d_scene);

  if (scene->transformSync)
    cudaDeviceSynchronize();

  TRACYCZONEEND(resetbbsKernelRun);

  TRACYCZONENC(transformKernelRun, "Transforms kernel", true, PROFILER_DARK_GREEN);

  // Launch a thread per vert
  size_t numTransformBlocks = (scene->pointsCount + threadsPerBlock - 1) / threadsPerBlock;
  transformKernel<<<numTransformBlocks, threadsPerBlock>>>(d_scene);

  if (scene->transformSync)
    cudaDeviceSynchronize();

  TRACYCZONEEND(transformKernelRun);

  TRACYCZONENC(normalsKernelRun, "Normals kernel", true, PROFILER_DARK_GREEN);

  // Launch a thread per triangle
  size_t numNormalBlocks = (scene->triangleNum + threadsPerBlock - 1) / threadsPerBlock;
  normalComputeKernel<<<numNormalBlocks, threadsPerBlock>>>(d_scene);

  if (scene->transformSync)
    cudaDeviceSynchronize();

  TRACYCZONEEND(normalsKernelRun);
}