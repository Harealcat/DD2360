#include <stdio.h>
#include <stdlib.h>


#define ARRAY_SIZE 10000
#define TBB 256

__global__ void saxpyGPU(int a, int n, float *x, float *y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n) y[i] = a*x[i] + y[i];
}

void saxpyCPU(int a, float *x, float *y){
	for(int i=0;i<ARRAY_SIZE;i++){
		y[i] = a*x[i] + y[i];
	}
}

void random_array(float *x){
   	for(int i=0;i<ARRAY_SIZE;i++){
     	x[i]=rand()%100;   
    }
}

void print_array(float *x){
	for(int i=0;i<ARRAY_SIZE;i++){
     	printf("%f;", x[i]);  
    }
}

int main(int argc, char **argv){
	int grid = (ARRAY_SIZE+(TBB-(ARRAY_SIZE%TBB)))/TBB;
	int a = rand()%100;

	float *x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
	float *w = (float *)malloc(sizeof(float) * ARRAY_SIZE) ;
	random_array(x);
	memcpy(w, x, sizeof(float) * ARRAY_SIZE);

	float *y = (float *)malloc(sizeof(float) * ARRAY_SIZE);
	float *z = (float *)malloc(sizeof(float) * ARRAY_SIZE);
	random_array(y);
	memcpy(z, y, sizeof(float) * ARRAY_SIZE);

	float *d_x = NULL;
	float *d_y = NULL;

	cudaMalloc(&d_x, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_y, ARRAY_SIZE * sizeof(float));

	cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	saxpyGPU<<<grid,TBB>>>(a, ARRAY_SIZE, d_x, d_y);

	cudaMemcpy(x, d_x, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	printf("Computing SAXPY on the GPU... Done!\n");

	saxpyCPU(a, w, z);
	printf("Computing SAXPY on the CPU... Done!\n");

	printf("Comparing the output for each implementation...\n");

	int n = memcmp(y, z, sizeof(float) * ARRAY_SIZE);
	if (n==0){
		printf("Correct!\n");
	} 
	else{
		printf("Not Correct!\n");
	}

	free(x);
	free(y);
	free(w);
	free(z);
	cudaFree(d_x);
	cudaFree(d_y);


	return 0;
}