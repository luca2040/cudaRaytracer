#include "TransformKernel.cuh"

#include "../../../../math/cuda/CudaMath.cuh"

__global__ void resetAABBsKernel(Scene *scene)
{
  size_t objIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (objIdx >= scene->sceneobjectsNum)
    return;

  SceneObject *parentObject = &scene->d_sceneobjects[objIdx];
  AABB *currentAABB = &parentObject->boundingBox;

  currentAABB->l = make_float3_L(INFINITY, INFINITY, INFINITY);
  currentAABB->h = make_float3_L(-INFINITY, -INFINITY, -INFINITY);
}

__global__ void transformKernel(Scene *scene)
{
  size_t vertIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertIdx >= scene->pointsCount)
    return;

  size_t objIdx = scene->d_pointToObjIdxTable[vertIdx];
  mat4x4 vertexMat = scene->d_transformMatrices[objIdx];

  float3_L currentVert = scene->d_pointarray[vertIdx];
  float4_L currentVert4 = make_float4_L(currentVert.x, currentVert.y, currentVert.z, 1.0f);

  float4_L transformedVert4 = currentVert4 * vertexMat;
  float3_L transformedVert = make_float3_L(transformedVert4.x, transformedVert4.y, transformedVert4.z);

  scene->d_trsfrmdpoints[vertIdx] = transformedVert;

  SceneObject *parentObject = &scene->d_sceneobjects[objIdx];
  AABB *currentAABB = &parentObject->boundingBox;

  atomicMinFloat(&currentAABB->l.x, transformedVert.x);
  atomicMinFloat(&currentAABB->l.y, transformedVert.y);
  atomicMinFloat(&currentAABB->l.z, transformedVert.z);

  atomicMaxFloat(&currentAABB->h.x, transformedVert.x);
  atomicMaxFloat(&currentAABB->h.y, transformedVert.y);
  atomicMaxFloat(&currentAABB->h.z, transformedVert.z);
}

__global__ void normalComputeKernel(Scene *scene)
{
  size_t triIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (triIdx >= scene->triangleNum)
    return;

  triangleidx &currentTri = scene->d_triangles[triIdx];

  float3_L v1 = scene->d_trsfrmdpoints[currentTri.v1];
  float3_L v2 = scene->d_trsfrmdpoints[currentTri.v2];
  float3_L v3 = scene->d_trsfrmdpoints[currentTri.v3];

  float3_L side1 = v2 - v1;
  float3_L side2 = v3 - v1;

  currentTri.normal = normalize3_cuda(cross3_cuda(side1, side2));
}