#include "Functions.h"
#include <malloc.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>

// CUDA error checking functions
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		system("PAUSE");
		if (abort) exit(code);
	}
}

#define iDivUp(a,b) (((int)(a) % (int)(b) != 0) ? (((int)(a) /(int) (b)) + 1) : ((int)(a) /(int) (b)))

// Rescaling function to output images on [0,1] interval
Mat rescale(Mat input, float c, float d){

	float a = input.at<float>(0,0);
	float b = input.at<float>(0,0);

	for (int i=0;i<input.cols;i++){
		for (int j=0;j<input.rows;j++){
			if (input.at<float>(j,i) < a){
				a = input.at<float>(j,i);
			}
			if (input.at<float>(j,i) > b){
				b = input.at<float>(j,i);
			}
		}
	}
	cout << " Before scaling: min = " << a << ", max =  " << b << "." << endl;
	for (int i=0;i<input.cols;i++){
		for (int j=0;j<input.rows;j++){
			if (b != a)
				input.at<float>(j,i) = input.at<float>(j,i)*(d-c)/(b-a) + (b*c-a*d)/(b-a);	
			else
				input.at<float>(j,i) = 1; 
		}
	}

	a = input.at<float>(0,0);
	b = input.at<float>(0,0);

	for (int i=0;i<input.cols;i++){
		for (int j=0;j<input.rows;j++){
			if (input.at<float>(j,i) < a){
				a = input.at<float>(j,i);
			}
			if (input.at<float>(j,i) > b){
				b = input.at<float>(j,i);
			}
		}
	}
	cout << " After scaling: min = " << a << ", max =  " << b << "." << endl;
	return input;
}
// Matlab is column-major, C is row-major
void columnOrderToRowOrder(Mat input, float* output, int ncols, int nrows){

	for (int i = 0; i<nrows; i++){
		for (int j = 0; j<ncols; j++){
			output[j + i*ncols] = input.at<float>(i,j);
		}
	}

}
void columnOrderToRowOrder_Mat(Mat input, float* output, int nrows, int ncols){

	for (int i = 0; i<nrows; i++){
		for (int j = 0; j<ncols; j++){
			input.at<float>(i,j)= output[j + i*ncols];
		}
	}
}

//This function will read any binary data file, assuming that it contains contiguous 4-byte floating point numbers, 
//and store them into an array. It will return a pointer to the memory block containing this data. 

float* read_data (const char *inputfile){
	printf("%==================== \n read_data: \n");

	//Associate inputfile with a stream via ifptr, open file as binary data.
	FILE *ifPtr;
	ifPtr=fopen(inputfile,"rb");

	//Report success or display error message. 
	if(ifPtr==NULL){
		perror("	Error reading inputfile ");
		return NULL;
	}
	else{
		printf("	Inputfile '%s' opened. \n",inputfile);

		//Determine file size
		fseek(ifPtr,0,SEEK_END);
		int ifSize=ftell(ifPtr);
		rewind(ifPtr);
		printf("	Filesize is %i bytes. \n",ifSize);

		//Initialize data pointer and allocate a sufficient amount of memory
		float* DataPtr= (float*) malloc(ifSize);

		if (DataPtr==NULL){
			perror("	Problem allocating memory.");
			printf("\n File will not be read. \n \n");
		}

		//If memory allocation went well, continue to read file.
		else{
			printf("	%i bytes of memory succesfully allocated at %p \n",ifSize,DataPtr);
			
			//Read binary data into a memoryblock of size ifSize, pointed to by Data

			//Mogelijk probleem: op andere systemen vari�ert sizeof(float) maar de data zal altijd 4 bytes/element zijn.
			int Length=ifSize/(sizeof(float));
			int Elements_Read=fread(DataPtr,4,Length,ifPtr);
			printf("	%i elements out of %i read. \n \n",Elements_Read,Length);
		}
		return DataPtr;
		fclose(ifPtr);
		free(ifPtr);
		free(DataPtr);
	}
}

//This function will copy the Data to the device memory for parallel manipulation
float* copy_data_to_device(float* Data,const int image_size){
	printf("%==================== \n Allocating GPU memory & copying data: \n");
	if(Data!=NULL){
		float* d_Data=NULL;
		size_t allocation_size=image_size*image_size*sizeof(float);

		//Allocate memory on device
		cudaError_t err=cudaSuccess;
		err=cudaMalloc((void**)&d_Data,allocation_size);
		if(err!=cudaSuccess){
			printf("	Failed to allocate GPU memory: %s. \n",cudaGetErrorString(err));
			return NULL;
		}
		else{
			printf("	GPU memory succesfully allocated at %p.\n",d_Data);
			//Copy data
			err=cudaMemcpy(d_Data,Data,allocation_size,cudaMemcpyHostToDevice);
			if (err!=cudaSuccess){
				printf("	Failed to copy data from %p to %p: %s. \n",Data,d_Data,cudaGetErrorString(err));
				return NULL;
			}
			else{
				printf("	Data successfully copied to GPU at %p. \n",d_Data);
				return d_Data;
			}
		}
	}
	else{
		printf("copy_data_to_device: Data is a NULL pointer.");
		return NULL;
	}
	
}

//Checks validity of input device pointer and copies it to host memory.
float* copy_data_to_host(const float* d_Data,const int image_size){
	size_t memory_size=sizeof(float)*image_size*image_size;
	float* h_Output=(float*)malloc(memory_size);
	if(h_Output!=NULL || d_Data!=NULL){
		cudaError err=cudaSuccess;
		err=cudaMemcpy((void**)h_Output,(void*)d_Data,memory_size,cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess){
			printf("copy_data_to_host: cuda: %s \n",cudaGetErrorString(err));
			return NULL;
		}
		else{
			printf("copy_data_to_host: data successfully copied \n");
			return h_Output;
		}
	}
	else{
		printf("copy_data_to_host: Failed to allocate host memory or d_Data pointer is NULL. \n");
		return NULL;
	}
}

//Gamma-function: a "modulo 2Pi"-function
float gamma(float& input){
	return fmodf(input,TWOPI);
}

//Single pixel-unwrapper, unwraps pixel 2 w.r.t. pixel 1, passing by value, as we don't want to change the input arguments. 
float unwrap(float pixel_value1,float pixel_value2)
{
	float delta=pixel_value2-pixel_value1;
	while(delta>=TWOPI){
			pixel_value2=pixel_value2-TWOPI;
			delta=delta-TWOPI;
	}
	while(delta<=-TWOPI){
		pixel_value2=pixel_value2+TWOPI;
		delta=delta+TWOPI;
	}
	float result=pixel_value2;
	return result;
}

//CPU-unwrapper Itoh algorithm

float* unwrap_cpu(float* data, int& n){
	float * Output;
	Output=(float*)malloc(sizeof(float)*n*n);
		if (Output != NULL && data != NULL){
			for (int column=0; column < n; column++){
				Output[n*column]=data[n*column];
				float deltaref=0;
				for (int row=0; row <= n; row++){
					float delta=data[n*column+row+1]-data[n*column+row];
					if(std::abs(delta)>=TWOPI){
						deltaref=deltaref-delta;
					}
					Output[n*column+row+1]=data[n*column+row+1]+deltaref;
				}
			}
			//Unwrap in de andere richting

			for(int row=0; row < n ; row++){
				float deltaref=0;
				for(int column=0 ; column < n; column++){
					float delta=Output[n*column+row+n]-Output[n*column+row];
					if(std::abs(delta)>=TWOPI){
						deltaref=deltaref-delta;
					}
					Output[n*column+row]=Output[n*column+row+n]+deltaref;
				}
			}


		}
		else{
			printf("unwrap_cpu: Failed to allocate Output memory or input pointer was NULL. \n");
			return NULL;
		}
	printf("unwrap_cpu: Finished unwrapping. \n");
	return Output;
}

//Export data
void export_data(const char* inputfile, float* data, int& elements_amount){
	if(data!=NULL){
		FILE* OfPtr;
		OfPtr=fopen(inputfile,"wb+");
		if(OfPtr!=NULL){
		int elements_written=fwrite(data,sizeof(float),elements_amount,OfPtr);
		printf("export_data: %i elements successfully written to Output.bin \n", elements_written);
		fclose(OfPtr);
		}
	}
	else
		printf("export_data: input pointer is NULL. \n");
}
