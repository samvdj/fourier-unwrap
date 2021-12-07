#include "Functions.h"
// General headers
#include "stdafx.h"
#include <stdio.h>  
#include <iostream>
// OpenCV headers
#include "cv.h" 
#include <highgui.h>
// CUDA headers
#include "cuda_runtime.h"
#include "cuda.h"
#include <cufft.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include "cuda_profiler_api.h"

using namespace cv;
using namespace std;
float pi = 3.1415926;
unsigned int ncols = 640;
unsigned int nrows = 480;

#define kernel_width 3
#define kernel_radius kernel_width/2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + kernel_width - 1)
#define pi 3.141592

// GPU: y = sin(x)
__global__ void get_sin(float *d_sin, float *input, int ncols, int nrows) 
{ 
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset;

	if(addr.x <ncols && addr.y < nrows){
		offset = addr.x + ncols*addr.y;
		d_sin[offset] = sin(input[offset]);
	} 
} 
// GPU: y = cos(x)
__global__ void get_cos(float *d_cos, float *input, int ncols, int nrows) 
{ 
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset;

	if(addr.x <ncols && addr.y < nrows){
		offset = addr.x + ncols*addr.y;
		d_cos[offset] = cos(input[offset]);
	} 
} 
// GPU: y = x.*(qx²+qy²)
__global__ void multiply_normalization_coefficients(float *input, int ncols, int nrows) 
{ 
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset;

	if(addr.x <ncols && addr.y < nrows){
		offset = addr.x + ncols*addr.y;
		input[offset] = input[offset]*(addr.x*addr.x + addr.y*addr.y);

		if (addr.x == 0 && addr.y == 0)
			input[offset] = 1;
	} 
}

// GPU: C = A.*B
__global__ void elementwise_matrixmul(float *A, float *B, float *C, int ncols, int nrows) 
{ 
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset;

	if(addr.x <ncols && addr.y < nrows){
		offset = addr.x + ncols*addr.y;

		C[offset] = A[offset]*B[offset];
	} 
}

// GPU: C = A + beta*B;
__global__ void elementwise_matrixaddmul(float *A, float *B, float *C, float beta, int ncols, int nrows) 
{ 
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset;

	if(addr.x <ncols && addr.y < nrows){
		offset = addr.x + ncols*addr.y;

		C[offset] = A[offset] + beta*B[offset];
	} 
}

// GPU: phase+2*pi*round((soln - phase)/(2*pi)))
__global__ void final_unwrap(float *soln, float *phase, float *d_mask, int ncols, int nrows) 
{ 
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset;

	if(addr.x <ncols && addr.y < nrows){
		offset = addr.x + ncols*addr.y;
	if (d_mask[offset] != 0){
	soln[offset] = soln[offset];
	}
	else{
    soln[offset] = 0;
	} 
}
}
// GPU: y = x./(qx²+qy²) with singular point at qx = qy = 0
__global__ void divide_normalization_coefficients(float *input, float *n, int ncols, int nrows) 
{ 
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset;

	if(addr.x <ncols && addr.y < nrows){
		offset = addr.x + ncols*addr.y;

		if (addr.x == 0 && addr.y == 0){
			input[offset] = 0.0;
		}
		else{
			input[offset] = input[offset]/n[offset];		
		}
	} 
}
// GPU: y = cos(x)*lapl(sin(x)) - sin(x)*lapl(cos(x))
__global__ void get_b(float *rarray, float *d_cos, float *d_DCT_result1, float *d_sin, float *d_DCT_result2, int ncols, int nrows) 
{ 
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset;

	if(addr.x <ncols && addr.y < nrows){
		offset = addr.x + ncols*addr.y;
		rarray[offset] = d_cos[offset]*d_DCT_result1[offset] - d_sin[offset]*d_DCT_result2[offset];
	} 
}
// GPU: actual adding/subtracting of n*2pi
__global__ void unwrap(float *d_A, float *d_phi_accent, int ncols, int nrows) 
{ 
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset;

	if(addr.x <ncols && addr.y < nrows){
		offset = addr.x + ncols*addr.y;
		d_A[offset] = d_A[offset] + 2*3.14159265*roundf((d_phi_accent[offset] - d_A[offset])/(2*3.14159265));

	} 
}




// Naive implementation (no shared memory)
__global__ void weighted_laplacian(float *input, float* mask, float* output, int ncols, int nrows) {
   
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int k, k1, k2, k3, k4, w1, w2, w3, w4;
	float r1, r2, r3, r4;

	if(addr.x <ncols && addr.y < nrows){
		// addr.x = i   and   addr.y = j
		k = addr.x + ncols*addr.y;
		k1 = (addr.x < ncols - 1) ? k + 1 : k - 1;
		k2 = (addr.x > 0) ? k - 1 : k + 1;
        k3 = (addr.y < nrows-1) ? k + ncols : k - ncols;
        k4 = (addr.y > 0) ? k - ncols : k + ncols;

		    w1 =  min(mask[k], mask[k1]);
            w2 =  min(mask[k], mask[k2]);
            w3 =  min(mask[k], mask[k3]);
            w4 =  min(mask[k], mask[k4]);

			r1 = input[k] - input[k1];
			r2 = input[k] - input[k2];
			r3 = input[k] - input[k3];
			r4 = input[k] - input[k4];

			if (r1 > pi) r1 = r1 - 2*pi;
			if (r1 < -pi) r1 = r1 + 2*pi;
			if (r2 > pi) r2 = r2 - 2*pi;
			if (r2 < -pi) r2 = r2 + 2*pi;
			if (r3 > pi) r3 = r3 - 2*pi;
			if (r3 < -pi) r3 = r3 + 2*pi;
			if (r4 > pi) r4 = r4 - 2*pi;
			if (r4 < -pi) r4 = r4 + 2*pi;

		    output[k] = w1*r1 + w2*r2 + w3*r3 + w4*r4;  
	} 
   
}

// Naive implementation (no shared memory)
__global__ void Qp(float *input, float* mask, float* output, int ncols, int nrows) {
   
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int k, k1, k2, k3, k4, w1, w2, w3, w4;
	float r1, r2, r3, r4;

	if(addr.x <ncols && addr.y < nrows){
		// addr.x = i   and   addr.y = j
		k = addr.x + ncols*addr.y;
		k1 = (addr.x < ncols - 1) ? k + 1 : k - 1;
		k2 = (addr.x > 0) ? k - 1 : k + 1;
        k3 = (addr.y < nrows-1) ? k + ncols : k - ncols;
        k4 = (addr.y > 0) ? k - ncols : k + ncols;

		    w1 =  min(mask[k], mask[k1]);
            w2 =  min(mask[k], mask[k2]);
            w3 =  min(mask[k], mask[k3]);
            w4 =  min(mask[k], mask[k4]);

		output[k] = (w1+w2+w3+w4)*input[k] - (w1*input[k1] + w2*input[k2] + w3*input[k3] + w4*input[k4]); 
	} 
   
}

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size excceed the upbound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = MIN(maxBlocks, blocks);
    }
}
template <unsigned int blockSize,bool nIsPow2>
__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n)
{
    
	extern __shared__ float sdata[];
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

   
    switch (threads)
                {
                    case 512:
                        reduce6<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 256:
                        reduce6<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 128:
                        reduce6<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 64:
                        reduce6<64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 32:
                        reduce6<32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 16:
                        reduce6<16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  8:
                        reduce6<8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  4:
                        reduce6<4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  2:
                        reduce6<2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  1:
                        reduce6<1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;
                }

}

     
float totalReduce(int  n,
                  int  numThreads,
                  int  numBlocks,                
                  float *h_odata,
                  float *d_idata,
                  float *d_odata)
{
    float gpu_result = 0;
    bool needReadBack = true;

        cudaDeviceSynchronize();

        // execute the kernel
        reduce(n, numThreads, numBlocks, d_idata, d_odata);

            // sum partial sums from each block on CPU
            // copy result from device to host
            cudaMemcpy(h_odata, d_odata, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);

            for (int i=0; i<numBlocks; i++)
            {
                gpu_result += h_odata[i];
            }

    return gpu_result;
}
int main(int argc, char** argv)
{
	// Load input data 
	float* h_A = read_data("Image_Wrapped.bin"); // Wrapped input image "A"
	float* h_mask = read_data("mask.bin"); // Mask
	float* h_DCT_XL = read_data("DCT_XL.bin"); // Pre-constructed transformation matrix coefficients
	float* h_DCT_XR = read_data("DCT_XR.bin");
	float* h_IDCT_XL = read_data("IDCT_XL.bin");
	float* h_IDCT_XR = read_data("IDCT_XR.bin");
	float* h_n = read_data("n.bin");

	//Convert input image to Mat format to display
	Mat h_A_mat = Mat(nrows, ncols, CV_32F);
	for(int i=0; i<ncols; i++){
		for(int j=0; j<nrows; j++)
		{
			h_A_mat.at<float>(j,i) = h_A[j + nrows*i];
		}
	}
	
	//Convert input image to Mat format to display
	Mat h_mask_mat = Mat(nrows, ncols, CV_32F);
	for(int i=0; i<ncols; i++){
		for(int j=0; j<nrows; j++)
		{
			h_mask_mat.at<float>(j,i) = h_mask[j + nrows*i];
		}
	}

	//Convert DCT_XL to Mat format to display
	Mat h_DCT_XL_mat = Mat(nrows, nrows, CV_32F);
	for(int i=0; i<nrows; i++){
		for(int j=0; j<nrows; j++)
		{
			h_DCT_XL_mat.at<float>(j,i) = h_DCT_XL[j + nrows*i];
		}
	}

	//Convert DCT_XR to Mat format to display
	Mat h_DCT_XR_mat = Mat(ncols, ncols, CV_32F);
	for(int i=0; i<ncols; i++){
		for(int j=0; j<ncols; j++)
		{
			h_DCT_XR_mat.at<float>(j,i) = h_DCT_XR[j + ncols*i];
		}
	}

	//Convert IDCT_XL to Mat format to display
	Mat h_IDCT_XL_mat = Mat(nrows, nrows, CV_32F);
	for(int i=0; i<nrows; i++){
		for(int j=0; j<nrows; j++)
		{
			h_IDCT_XL_mat.at<float>(j,i) = h_IDCT_XL[j + nrows*i];
		}
	}

	//Convert DCT_XR to Mat format to display
	Mat h_IDCT_XR_mat = Mat(ncols, ncols, CV_32F);
	for(int i=0; i<ncols; i++){
		for(int j=0; j<ncols; j++)
		{
			h_IDCT_XR_mat.at<float>(j,i) = h_IDCT_XR[j + ncols*i];
		}
	}
		
	//Convert n to Mat format to display
	Mat h_n_mat = Mat(nrows, ncols, CV_32F);
	for(int i=0; i<ncols; i++){
		for(int j=0; j<nrows; j++)
		{
			h_n_mat.at<float>(j,i) = h_n[j + nrows*i];
		}
	}

	//Data was read from Matlab (Column Order) into C/C++ (Row Order)
	columnOrderToRowOrder(h_A_mat, h_A, ncols, nrows);
	columnOrderToRowOrder(h_mask_mat, h_mask, ncols, nrows);
	columnOrderToRowOrder(h_DCT_XR_mat, h_DCT_XR, ncols, ncols);
	columnOrderToRowOrder(h_DCT_XL_mat, h_DCT_XL, nrows, nrows);
	columnOrderToRowOrder(h_IDCT_XR_mat, h_IDCT_XR, ncols, ncols);
	columnOrderToRowOrder(h_IDCT_XL_mat, h_IDCT_XL, nrows, nrows);
	columnOrderToRowOrder(h_n_mat, h_n, ncols, nrows);

	//Rescale data to [0,1] for visualization
	h_A_mat = rescale(h_A_mat,0,1);
	h_mask_mat = rescale(h_mask_mat,0,1);	
	h_DCT_XL_mat = rescale(h_DCT_XL_mat,0,1);
	h_DCT_XR_mat = rescale(h_DCT_XR_mat,0,1);
	h_IDCT_XL_mat = rescale(h_IDCT_XL_mat,0,1);
	h_IDCT_XR_mat = rescale(h_IDCT_XR_mat,0,1);

	// Copy DCT_XL, DCT_XR, IDCT_XL, DCT_XR and A to GPU memory
	float* d_A;
	gpuErrchk(cudaMalloc(&d_A,sizeof(float)*ncols*nrows));
	float* d_mask;
	gpuErrchk(cudaMalloc(&d_mask,sizeof(float)*ncols*nrows));
	float* d_DCT_XL;
	gpuErrchk(cudaMalloc(&d_DCT_XL,sizeof(float)*nrows*nrows));
	gpuErrchk(cudaMemcpy(d_DCT_XL, h_DCT_XL, sizeof(float)*nrows*nrows, cudaMemcpyHostToDevice));
	float* d_DCT_XR;
	gpuErrchk(cudaMalloc(&d_DCT_XR,sizeof(float)*ncols*ncols));
	gpuErrchk(cudaMemcpy(d_DCT_XR, h_DCT_XR, sizeof(float)*ncols*ncols, cudaMemcpyHostToDevice));
	float* d_IDCT_XL;
	gpuErrchk(cudaMalloc(&d_IDCT_XL,sizeof(float)*nrows*nrows));
	gpuErrchk(cudaMemcpy(d_IDCT_XL, h_IDCT_XL, sizeof(float)*nrows*nrows, cudaMemcpyHostToDevice));
	float* d_IDCT_XR;
	gpuErrchk(cudaMalloc(&d_IDCT_XR,sizeof(float)*ncols*ncols));
	gpuErrchk(cudaMemcpy(d_IDCT_XR, h_IDCT_XR, sizeof(float)*ncols*ncols, cudaMemcpyHostToDevice));
	float* d_n;
	gpuErrchk(cudaMalloc(&d_n,sizeof(float)*ncols*nrows));
	gpuErrchk(cudaMemcpy(d_n, h_n, sizeof(float)*ncols*nrows, cudaMemcpyHostToDevice));

	// Prepare pointers for temporary and final results	
	float* d_temp;
	gpuErrchk(cudaMalloc(&d_temp,sizeof(float)*nrows*ncols));
	float* c;
	gpuErrchk(cudaMalloc(&c,sizeof(float)*nrows*ncols));
	float* rarray;
	gpuErrchk(cudaMalloc(&rarray,sizeof(float)*nrows*ncols));
	float* zarray;
	gpuErrchk(cudaMalloc(&zarray,sizeof(float)*nrows*ncols));
    	float* parray;
	gpuErrchk(cudaMalloc(&parray,sizeof(float)*nrows*ncols));
    	float* d_soln;
	gpuErrchk(cudaMalloc(&d_soln,sizeof(float)*nrows*ncols));
	float* h_soln = (float*)malloc(sizeof(float)*nrows*ncols);
	for (int i = 0; i < nrows*ncols; i++){ h_soln[i] = 0; };
	gpuErrchk(cudaMemcpy(d_soln, h_soln, sizeof(float)*ncols*nrows, cudaMemcpyHostToDevice));


	float* h_final_result = (float*)malloc(sizeof(float)*nrows*ncols);

	// Setup device
	cudaDeviceProp deviceProp;
	cudaError_t error;
	int devID = 0;

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
		system("PAUSE"); exit(EXIT_FAILURE);
	}

	// Use a larger block size for Fermi and above
	int block_size = (deviceProp.major < 2) ? 16 : 32;

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(ncols / threads.x, nrows / threads.y);

	// Setup CUBLAS
	cublasHandle_t handle;
	cublasStatus_t ret;
	ret = cublasCreate(&handle);
	const float alpha = 1.0f;
	const float beta  = 0.0f;

	// Perform warmup operation with cublas (result = A*XR)
	{
		//ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncols, nrows, ncols, &alpha, d_DCT_XR, ncols, d_temp, ncols, &beta, rarray, ncols);
		//ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncols, nrows, nrows, &alpha, rarray, ncols, d_DCT_XL, nrows, &beta, d_temp, ncols);
	}


		int numBlocks = 0;
        	int numThreads = 0;
		int maxBlocks = 64;
		int maxThreads = 256;


		getNumBlocksAndThreads(6, nrows*ncols, maxBlocks, maxThreads, numBlocks, numThreads);
		cout << "numBlocks = " << numBlocks << endl;
		cout << "numThreads = " << numThreads << endl;
		
		// allocate mem for the result on host side
        float *h_odata = (float *) malloc(numBlocks*sizeof(float));

		 // allocate device memory and data
     
        float *d_odata = NULL;

        
        cudaMalloc((void **) &d_odata, numBlocks*sizeof(float));


	// Create and start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//int betasum = 1;
	//int betasum_prev = 1;




	int nIter = 100;
	int max_iter = 2;
	

	cout << endl << "Start computing results using CUBLAS..." << endl;

	// Record the start event
	cudaEventRecord(start, 0);

	// Loop over nIter iterations of the complete unwrapping algorithm
	for (int j = 0; j < nIter; j++)
	{
	gpuErrchk(cudaMemcpy(d_mask, h_mask, sizeof(float)*ncols*nrows, cudaMemcpyHostToDevice)); // *Copy input data here or later (see below) (timing)
	gpuErrchk(cudaMemcpy(d_A, h_A, sizeof(float)*ncols*nrows, cudaMemcpyHostToDevice)); // *Copy input data here or later (see below) (timing)
		
	
	cudaProfilerStart();


	
	gpuErrchk(cudaMemcpy(d_soln, h_soln, sizeof(float)*ncols*nrows, cudaMemcpyHostToDevice));
    	float betaprev = 1;
	
	
	// 1. Weighted Laplacian, only needed once for all iterations
	dim3 dimGrid(ceil((float)ncols/TILE_WIDTH), ceil((float)nrows/TILE_WIDTH));
   	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
   	weighted_laplacian<<<dimGrid, dimBlock>>>(d_A, d_mask, rarray, ncols, nrows);
	

	// Phase unwrapping loop
	for (int iloop = 0; iloop < max_iter; iloop++){
		// 2. zarray = rarray
		gpuErrchk(cudaMemcpy(zarray, rarray, sizeof(float)*nrows*ncols, cudaMemcpyDeviceToDevice));
		   
		// 3. c = dct2(zarray)
		ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncols, nrows, ncols, &alpha, d_DCT_XR, ncols, zarray, ncols, &beta, d_temp, ncols);
		ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncols, nrows, nrows, &alpha, d_temp, ncols, d_DCT_XL, nrows, &beta, c, ncols);

		// 4. normalization c/n
		divide_normalization_coefficients<<<dimGrid, dimBlock>>>(c,d_n,ncols,nrows);

		// 5. zarray = idct2(d)
		ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncols, nrows, ncols, &alpha, d_IDCT_XR, ncols, c, ncols, &beta, d_temp, ncols);
		ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncols, nrows, nrows, &alpha, d_temp, ncols, d_IDCT_XL, nrows, &beta, zarray, ncols);

		// 6. updates
		
		// beta = SUM(rarray.*zarray)
     		elementwise_matrixmul<<<dimGrid, dimBlock>>>(rarray,zarray,d_temp,ncols,nrows);
		float beta = totalReduce(nrows*ncols, numThreads, numBlocks, h_odata, d_temp, d_odata);
		
		if (iloop == 0)
		{
			// parray = zarray
			gpuErrchk(cudaMemcpy(parray, zarray, sizeof(float)*nrows*ncols, cudaMemcpyDeviceToDevice));
		
		}
		else
		{
			// parray = zarray + beta/betaprev*parray;
			float betatemp = beta/betaprev;
			elementwise_matrixaddmul<<<dimGrid, dimBlock>>>(zarray, parray,parray,betatemp,ncols,nrows);
		}

		
        //elementwise_matrixmul<<<dimGrid, dimBlock>>>(parray,d_mask,parray,ncols,nrows);
        betaprev = beta;

		// 7. calculate Qp
		Qp<<<dimGrid, dimBlock>>>(parray,d_mask,zarray,ncols,nrows);
		
		// 8. last updates for this iteration
		elementwise_matrixmul<<<dimGrid, dimBlock>>>(zarray,parray,d_temp,ncols,nrows);
		float alpha = totalReduce(nrows*ncols, numThreads, numBlocks, h_odata, d_temp, d_odata);
		alpha = beta/alpha;
		//cout << "alpha = " << alpha << endl;
		//cout << "beta = " << beta << endl;
		//rarray = rarray - alpha*zarray;
		elementwise_matrixaddmul<<<dimGrid, dimBlock>>>(rarray, zarray, rarray,-alpha,ncols,nrows);
			
		// 9. solution
		elementwise_matrixaddmul<<<dimGrid, dimBlock>>>(d_soln,parray,d_soln,alpha,ncols,nrows);
		//cudaThreadSynchronize();
		
		if (iloop == (max_iter - 1)){
		final_unwrap<<<dimGrid, dimBlock>>>(d_soln,d_A,d_mask,ncols,nrows);
		}
		

		

	}
	cudaMemcpy(h_final_result, d_soln, sizeof(float)*nrows*ncols, cudaMemcpyDeviceToHost); // **Copy output data here or earlier (see above) (timing)
	cudaThreadSynchronize();
	cudaProfilerStop();

	}
	
	// Record the stop event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cout << endl << "done..." << endl;

	// Wait for the stop event to complete
	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	// Compute and print the performance
	float msecPerIter = msecTotal / nIter;

	cout << endl << " " << nIter << " iterations in " << msecTotal << " ms ===> " << msecPerIter << " ms per Unwrap iteration" << endl;

	// Copy results back to CPU
	// cudaMemcpy(h_final_result, d_soln, sizeof(float)*nrows*ncols, cudaMemcpyDeviceToHost); // **Copy output data here or earlier (see above) (timing)

	 // Convert output image to Mat format to display
	Mat h_result_mat = Mat(nrows, ncols, CV_32F);
	for(int i=0; i<ncols; i++){
		for(int j=0; j<nrows; j++)
		{
			h_result_mat.at<float>(j,i) = h_final_result[j + nrows*i];
		}
	}
	columnOrderToRowOrder_Mat(h_result_mat, h_final_result, nrows, ncols);
	
	cout << h_result_mat.at<float>(100,100) << endl;
	h_result_mat = rescale(h_result_mat,0,1);

	// Display results
	namedWindow("Wrapped input image",1);
	imshow("Wrapped input image", h_A_mat ); 
	namedWindow("Mask",1);
	imshow("Mask", h_mask_mat ); 
	/*namedWindow("DCT_XL",1);
	imshow("DCT_XL", h_DCT_XL_mat ); 
	namedWindow("DCT_XR",1);
	imshow("DCT_XR", h_DCT_XR_mat ); 
	namedWindow("IDCT_XL",1);
	imshow("IDCT_XL", h_IDCT_XL_mat ); 
	namedWindow("IDCT_XR",1);
	imshow("IDCT_XR", h_IDCT_XR_mat ); */ 	
	namedWindow("Unwrapped output image",1);
	imshow("Unwrapped output image", h_result_mat );

	waitKey(0);
	cvDestroyAllWindows();
	cout << "Successfully reached end of program" << endl; 
	system("PAUSE");
} 

