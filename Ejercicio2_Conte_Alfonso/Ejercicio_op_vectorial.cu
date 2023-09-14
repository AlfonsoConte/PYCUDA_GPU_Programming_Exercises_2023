#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
using namespace std;
static int M;
static int NBlocks;
static int N;


//CPU implementation
void cpu_C(float * A, float * B, float * C, int N, int Bsize)
{
	for (int k=0; k<NBlocks; k++)
	{
		int istart = k*Bsize;
  		int iend   = istart+Bsize;
  		for (int i=istart; i<iend;i++)
  		{
    			C[i]=0.0;
    			for (int j=istart; j<iend;j++)
      				C[i]+= fabs((i* B[j]-A[j]*A[j])/((i+2)*max(A[j],B[j])));
  		}
	}
}

//Kernel to determine vector C without Shared Memory
__global__ void C_GPU_Kernel(float * A, float * B, float * C, int N)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x; 
 	int start = blockIdx.x*blockDim.x; 

 	//Compute C[i] 
 	for(int j = start; j<start+blockDim.x; j++){
 		C[index] += fabs((index*B[j]-A[j]*A[j])/((index+2)*max(A[j],B[j])));
 	}
}

/*Kernel to determine vector C using Shared Memory.
The initialization of the shared memory will be made only by the first thread, such that all
the others thread of the block could use the shared memory.
The vector Sha Will contain the portion of A and B needed to compute C[id], respectivly in the first blockDim.x and 
in the second blockDim.x part.*/
__global__ void C_GPU_shm_Kernel( float * A, float * B, float * C, int N)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int start = blockIdx.x*blockDim.x; 
	int tid = threadIdx.x;

	extern __shared__ float sdata[];
	if(tid == 0){
		for(int i = 0; i<blockDim.x; i++){
			sdata[i] = A[start + i];
			sdata[i+blockDim.x] = B[start + i];
		}
	}
	__syncthreads();

 	for(int j = 0; j<blockDim.x; j++){
 		C[index] += fabs((index*sdata[blockDim.x + j]-sdata[j]*sdata[j])/((index+2)*max(sdata[j],sdata[blockDim.x + j])));}
}

//Kernel to compute the maximum of the vector C and do the reduction
__global__ void reduce_Max_Kernel(float * C, float * Maxs, int N)
{
	extern __shared__ float sdata[];
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	//Copying the values in the Sahred Memory
	sdata[tid] = ((index < N) ? C[index] : 0.0f);
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s>>=1){
	  if (tid < s)
	        sdata[tid]=max(sdata[tid],sdata[tid+s]);
	  __syncthreads();
	}
	if (tid == 0)
           Maxs[blockIdx.x] = sdata[0];
}

//Kernel to reduce C and compute D vector
__global__ void reduce_Sum_Kernel(float * C, float * D, int N)
{
	
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	extern __shared__ float sdata[];

	sdata[tid] = ((index<N) ? C[index] : 0.0f);
	__syncthreads();

	for (int s=blockDim.x/2; s>0; s>>=1){
		if (tid < s) {
		sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0){
		D[blockIdx.x] = sdata[0]/blockDim.x;}
}


//MAIN PROGRAM
int main(int argc, char *argv[])
{	
	if (argc != 3){ 
	  cout << "Uso: transformacion Num_bloques Tam_bloque  "<<endl;
	  return(0);
	}
	else{
	  NBlocks = atoi(argv[1]);
	  M= atoi(argv[2]);
	  N= M*NBlocks;
	}

	float * A;
	float * B;
	float * C;
	float * D;
	float * C_shared;
	float * A_GPU;
	float * B_GPU;
	float * C_GPU;
	float * C_g_shared;
	float * D_GPU;
	float * max_GPU;
	float * max_values;
	cudaError_t err;

	A = new float[N];
	B = new float[N];
	C = new float[N];
	D = new float[NBlocks];
	C_shared = new float[N];
	max_values = new float[NBlocks];

	//Initialization
	for (int i=0; i<N;i++)
	{
  		A[i] = (float) (1.5*(1+(5*i)%7)/(1+i%5));
  		B[i] = (float) (2.0*(2+i%5)/(1+i%7));
	}


	err=cudaMalloc((void **) &A_GPU, sizeof(float)*N);
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}
	err=cudaMalloc((void **) &B_GPU, sizeof(float)*N);
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}
	err=cudaMalloc((void **) &C_GPU, sizeof(float)*N);
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}
	err=cudaMalloc((void **) &C_g_shared, sizeof(float)*N);
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}
	err=cudaMalloc((void **) &D_GPU, sizeof(float)*NBlocks);
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}
	err=cudaMalloc((void **) &max_GPU, sizeof(float)*NBlocks); 
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}



	//Copying data from HOST to DEVICE 
	err=cudaMemcpy(A_GPU, A, sizeof(float)*N, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 
	err=cudaMemcpy(B_GPU, B, sizeof(float)*N, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 

	dim3 dimBlock(M);
	dim3 dimGrid(ceil((float(N)/(float)dimBlock.x)));

	double t1_gpu=clock();
	C_GPU_Kernel<<<dimGrid, dimBlock>>>(A_GPU, B_GPU, C_GPU, N);
	cudaDeviceSynchronize();
	double t2_gpu=clock();
	t2_gpu = (t2_gpu-t1_gpu)/CLOCKS_PER_SEC;

	
	double t1_cpu = clock();
	float * C1 = new float[N];
	cpu_C(A,B,C1,N, M); 
	double t2_cpu=clock();
	t2_cpu = (t2_cpu-t1_cpu)/CLOCKS_PER_SEC;


	double t1_gpu_shm=clock();
	int smemSize = 2*dimBlock.x*sizeof(float); //shared memory to store two "section" of A and B
	C_GPU_shm_Kernel<<<dimGrid, dimBlock, smemSize>>>(A_GPU, B_GPU, C_g_shared, N);
	cudaDeviceSynchronize();
	double t2_gpu_shm=clock();
	t2_gpu_shm = (t2_gpu_shm-t1_gpu_shm)/CLOCKS_PER_SEC;

	
	cout<<endl<<"> Time spent on CPU : "<<t2_cpu;
	cout<<endl<<"> Time spent on GPU : "<<t2_gpu;
	cout<<endl<<"> Time spent on GPU with shared memory: "<<t2_gpu_shm;
	err=cudaMemcpy(C, C_GPU, sizeof(float)*N,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 
	err=cudaMemcpy(C_shared, C_g_shared, sizeof(float)*N,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 


	int smemSize2 = M*sizeof(float);
	reduce_Max_Kernel<<<dimGrid, dimBlock, smemSize2>>>(C_g_shared, max_GPU, N);
	cudaDeviceSynchronize();
	err=cudaMemcpy(max_values, max_GPU, sizeof(float)*NBlocks, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 
	float final_max = 0.0f;
	//computing the last step of reduction
	for(int i = 0; i<NBlocks; i++) 
		final_max = max(final_max, max_values[i]);
	cout<<endl<<"> Maximum of C: "<<final_max<<endl;

	reduce_Sum_Kernel<<<dimGrid, dimBlock, smemSize2>>>(C_g_shared, D_GPU, N);
	cudaDeviceSynchronize();
	err=cudaMemcpy(D, D_GPU, sizeof(float)*NBlocks, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 


	delete(A);
	delete(B);
	delete(C);
	delete(C_shared);
	delete(D);
	delete(max_values);
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
	cudaFree(C_g_shared);
	cudaFree(D_GPU);
	cudaFree(max_GPU);

	return 0;
}
