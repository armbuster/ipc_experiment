#include <iostream>
#include "assert.h"


#define WORK_PER_THREAD 16

__global__ void saxpy_parallel(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	i *= WORK_PER_THREAD;
	
	if (i < n)
	{
		#pragma unroll
		for(int j=0; j<WORK_PER_THREAD; j++)
			y[i+j] = a * x[i+j] + y[i+j];
	}
}


void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }                         
}

int main()
{
	int N = 32 * 10000000;
	// allocate vectors on host
	int size = N * sizeof(float);
	float* h_x = (float*)malloc(size);
	float* h_y = (float*)malloc(size);
	

	// allocate device memory
	float* d_x; float* d_y;

	cudaMalloc((void**) &d_x, size);
	cudaMalloc((void**) &d_y, size);

	// put values in h_x and h_y

	for (int i = 0; i<N ;i++)
	{
		h_x[i]= (float) i;
		h_y[i]= (float) i;
	}

	cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

	// calculate number of blocks needed for N 
	int nblocks = ((N / WORK_PER_THREAD)+255)/256;

	// call 
	saxpy_parallel<<<nblocks,256>>>(N , a, d_x, d_y);
	
	// Copy results back from device memory to host memory
	// implicty waits for threads to excute
	cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

	for(int i = 0; i<N; i++) assert(h_y[i] == 2*i);

	// Check for any CUDA errors
  	checkCUDAError("cudaMemcpy calls");

	cudaFree(d_x);
	cudaFree(d_y);

	free(h_x);
	free(h_y);
	return 0;

}