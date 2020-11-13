#include <stdio.h>

__global__ void helloworld()
{
    int threadId = threadIdx.x;
    printf("Hello from the GPU! My threadId is %d\n", threadId);
}

int main(int argc, char** argv)
{
    dim3 grid(1); // 1 block in the grid
    dim3 block(256); // 256 threads per block

    helloworld << <grid, block >> > ();

    return 0;
}