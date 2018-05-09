#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

//#define cudaCheck(op) 						\\
//{								\\
//	int err;						\\
//	if (cudaSuccess != err = op) 				\\
//	{							\\
//		fprintf(stderr, "CUDA operation failed: %s\n",	\\
//				cudaGetErrorString(err));	\\
//		exit(EXIT_FAILURE);				\\
//	}							\\
//}

#define cudaCheck(op) op

__global__ void kernel(float *K, float *g1, float *M, int N, float k)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < N && j < N) M[i+j*N] = g1[N+i-j] * (K[i]+k) * (K[j]+k);
}

int main(int argc, char **argv)
{
	using namespace std::chrono;

	int N = 1000;
	if (argc == 2) {
		N = atoi(argv[1]);
	}
	printf("Using %d x %d\n", N, N); 

	float *M  = new float[N*N];
	float *K  = new float[N];
	float *g1 = new float[2*N];
	float k = 1.3;

	for (int i=0; i<N; i++)
		K[i]  = rand() / (float)RAND_MAX;
	for (int i=0; i<2*N; i++)
		g1[i] = rand() / (float)RAND_MAX;

	auto t0 = high_resolution_clock::now();

	float *d_M  = NULL;
	float *d_K  = NULL;
	float *d_g1 = NULL;

	cudaCheck(cudaMalloc((void**)&d_M,  sizeof(float)*N*N));
	cudaCheck(cudaMalloc((void**)&d_K,  sizeof(float)*N  ));
	cudaCheck(cudaMalloc((void**)&d_g1, sizeof(float)*2*N));

	cudaCheck(cudaMemcpy(d_M,   M, sizeof(float)*N*N, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_K,   K, sizeof(float)*N  , cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_g1, g1, sizeof(float)*2*N, cudaMemcpyHostToDevice));

	dim3 dimBlock(128); 
	dim3 dimGrid(N*N/dimBlock.x + (N*N % dimBlock.x ? 1 : 0));
	kernel<<<dimGrid,dimBlock>>>(K,g1,M,N,k);

	cudaCheck(cudaGetLastError());
	cudaCheck(cudaMemcpy(M, d_M, sizeof(float)*N*N, cudaMemcpyDeviceToHost));
	cudaCheck(cudaFree(d_M));
	cudaCheck(cudaFree(d_K));
	cudaCheck(cudaFree(d_g1));

	duration<float> t = high_resolution_clock::now() - t0;
	printf("GPU took %f seconds\n", t.count());

	delete[] M;
	delete[] K;
	delete[] g1;
	
	return 0;
}
