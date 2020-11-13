#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

typedef struct
{
   float3 position;
   float3 velocity;
} Particle;

__host__ __device__ void update_velocity(Particle *p, float3 *r){
	p->velocity.x = r->x;
	p->velocity.y = r->y;
	p->velocity.z = r->z;
}

__host__ __device__ void update_position(Particle *p){
	p->position.x = p->position.x + p->velocity.x;
	p->position.y = p->position.y + p->velocity.y;
	p->position.z = p->position.z + p->velocity.z;
}


__global__ void actionGPU(Particle *p, int N, int NI, float3 *r){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<N){
		for(int j=0;j<NI;j++){
			update_velocity(&p[i], &r[j]);
			update_position(&p[i]);
		}
	}
}

void actionCPU(Particle *p, float3 *r, int N, int NI){
	for(int i=0;i<N;i++){
		for(int j=0;j<NI;j++){
			update_velocity(&p[i], &r[j]);
			update_position(&p[i]);
		}
	}
}

void random_particles(Particle *p, int N){
	for(int i=0;i<N;i++){
		p[i].velocity.x = rand()%20;
		p[i].velocity.y = rand()%20;
		p[i].velocity.z = rand()%20;
		p[i].position.x = rand()%20;
		p[i].position.y = rand()%20;
		p[i].position.z = rand()%20;
	}
}

void random_velocity(float3 *r, int N){
	for(int i=0;i<N;i++){
		if (rand()%6 > 3){
			r[i].x = rand()%5;
			r[i].y = rand()%5;
			r[i].z = rand()%5;
		}
		else{
			r[i].x = -rand()%5;
			r[i].y = -rand()%5;
			r[i].z = -rand()%5;
		}
	}
}

int main(int argc, char **argv){
	int NUM_PARTICLES = atoi(argv[1]);
	int NUM_ITERATIONS = atoi(argv[2]);
	int TBB = atoi(argv[3]);
	LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    double interval;
	time_t t;
	srand((unsigned) time(&t));
	int grid = (NUM_PARTICLES+(TBB-(NUM_PARTICLES%TBB)))/TBB;
	float3 *r = (float3 *)malloc(sizeof(float3) * NUM_ITERATIONS);
	random_velocity(r, NUM_ITERATIONS);

	Particle *p = (Particle *)malloc(sizeof(Particle) * NUM_PARTICLES);
	Particle *q = (Particle *)malloc(sizeof(Particle) * NUM_PARTICLES);
	random_particles(p, NUM_PARTICLES);
	memcpy(q, p, sizeof(Particle) * NUM_PARTICLES);

	QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
	Particle *d_p = NULL;
	float3 *d_r = NULL;
	cudaMalloc(&d_p, sizeof(Particle) * NUM_PARTICLES);
	cudaMalloc(&d_r, sizeof(float3) * NUM_ITERATIONS);
	cudaMemcpy(d_p, p, sizeof(Particle) * NUM_PARTICLES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, r, sizeof(float3) * NUM_ITERATIONS, cudaMemcpyHostToDevice);

	actionGPU<<<grid,TBB>>>(d_p, NUM_PARTICLES, NUM_ITERATIONS, d_r);

	cudaMemcpy(p, d_p, sizeof(Particle) * NUM_PARTICLES, cudaMemcpyDeviceToHost);
	QueryPerformanceCounter(&end);
    interval = (double) (end.QuadPart - start.QuadPart) / frequency.QuadPart;
	printf("execution time for GPU : %f\n", interval);

	QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
	actionCPU(q, r, NUM_PARTICLES, NUM_ITERATIONS);
	QueryPerformanceCounter(&end);
    interval = (double) (end.QuadPart - start.QuadPart) / frequency.QuadPart;
	printf("execution time for CPU : %f\n", interval);

	printf("Comparing the output for each implementation...\n");
	int n = memcmp(p, q, sizeof(Particle) * NUM_PARTICLES);
	if (n==0){
		printf("Correct!\n");
	} 
	else{
		printf("Not Correct!\n");
	}
	
	free(p);
	free(q);
	cudaFree(d_p);

	return 0;
}