#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Devices: %d\n", devCount);
    return 0;
}