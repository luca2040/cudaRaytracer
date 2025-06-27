#include <iostream>
#include "test.cuh"

__global__ void printKernel()
{
  printf("Cuda Kernel test\n");
}

void testCUDA()
{
  printKernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}
