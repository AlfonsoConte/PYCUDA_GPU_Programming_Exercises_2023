#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

using namespace std;

#define blocksize_x 16
#define blocksize_y 16

//**************************************************************************

//**************************************************************************
__global__ void floyd_kernel(int * M, const int nverts, const int k) {
    int i= threadIdx.x + blockDim.x * blockIdx.x;
    int j= threadIdx.y + blockDim.y * blockIdx.y;
    int ij=i*nverts+j;
    if (i<nverts && j< nverts) {
    int Mij = M[i*nverts + j];
    if (i != j && i != k && j != k) {
	  int Mikj = M[i * nverts + k] + M[k * nverts + j];
    Mij = (Mij > Mikj) ? Mikj : Mij;
    M[ij] = Mij;}
  }
}


__global__ void reduceSum(int *d_V,int * V_tot,const int N)
{
		// Shared memory vector to store the data
		extern __shared__ int sdata[];
		// Compute global index i to access the vector d_V
		int tid = threadIdx.x;
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		// Load data into shared memory
		sdata[tid] = ((i < N) ? d_V[i] : 0.0f);
		__syncthreads();
		// Do reduction in shared memory
		for (int s=blockDim.x/2; s>0; s>>=1) {
			if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
		}
		// Write result for this block to global memory
		if (tid == 0) V_tot[blockIdx.x] = sdata[0];
}


//**************************************************************************
// ************  MAIN FUNCTION *********************************************
int main (int argc, char *argv[]) {

    double time, Tcpu, Tgpu;

    if (argc != 2) {
	    cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}	

    //Get GPU information
    int num_devices,devID;
    cudaDeviceProp props;
    cudaError_t err;

	err=cudaGetDeviceCount(&num_devices);
	if (err == cudaSuccess) { 
	    cout <<endl<< num_devices <<" CUDA-enabled  GPUs detected in this computer system"<<endl<<endl;
		cout<<"....................................................."<<endl<<endl;}	
	else 
	    { cerr << "ERROR detecting CUDA devices......" << endl; exit(-1);}
	    
	for (int i = 0; i < num_devices; i++) {
	    devID=i;
	    err = cudaGetDeviceProperties(&props, devID);
        cout<<"Device "<<devID<<": "<< props.name <<" with Compute Capability: "<<props.major<<"."<<props.minor<<endl<<endl;
        if (err != cudaSuccess) {
		  cerr << "ERROR getting CUDA devices" << endl;
	    }


	}
	devID = 0;    
        cout<<"Using Device "<<devID<<endl;
        cout<<"....................................................."<<endl<<endl;

	err = cudaSetDevice(devID); 
    if(err != cudaSuccess) {
		cerr << "ERROR setting CUDA device" <<devID<< endl;
	}

	// Declaration of the Graph object
	Graph G;
	
	// Read the Graph
	G.lee(argv[1]);

	//cout << "The input Graph:"<<endl;
	//G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;
	const int nverts2 = nverts * nverts;

	int *c_Out_M = new int[nverts2];
	int size = nverts2*sizeof(int);
	int * d_In_M = NULL;

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}
    // Get the integer 2D array for the dense graph
	int *A = G.Get_Matrix();

    //**************************************************************************
	// GPU phase
	//**************************************************************************
	
    time=clock();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 


	dim3 threadsPerBlock (blocksize_x,blocksize_y);
	dim3 numBlocks (ceil ((float)(nverts)/threadsPerBlock.x),ceil((float)(nverts)/threadsPerBlock.y));
    // Main Loop
	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
        // Kernel Launch
	    floyd_kernel<<<numBlocks,threadsPerBlock >>>(d_In_M, nverts, k);
	    err = cudaGetLastError();

	    if (err != cudaSuccess) {
	  	    fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	    exit(EXIT_FAILURE);
		}
	}
	err =cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 

	Tgpu=(clock()-time)/CLOCKS_PER_SEC;
	
	cout << "Time spent on GPU= " << Tgpu << endl << endl;

    //**************************************************************************
	// CPU phase
	//**************************************************************************

	time=clock();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < niters; k++) {
          kn = k * nverts;
	  for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++)
	       			if (i!=j && i!=k && j!=k){
			 	    inj = in + j;
			 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
	       }
	   }
	}
  
  Tcpu=(clock()-time)/CLOCKS_PER_SEC;
  cout << "Time spent on CPU= " << Tcpu << endl << endl;
  cout<<"....................................................."<<endl<<endl;

  cout << "Speedup TCPU/TGPU= " << Tcpu / Tgpu << endl;
  cout<<"....................................................."<<endl<<endl;

  
  bool errors=false;
  // Error Checking (CPU vs. GPU)
  for(int i = 0; i < nverts; i++)
    for(int j = 0; j < nverts; j++)
       if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
         {cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;
		  errors=true;
		 }

	//SUM REDUCTION TO CALCULATE THE MEAN OF ALL THE SHORTEST PATH
		dim3 threadsPerBlockMean(256,1);
		dim3 numBlocksMean( ceil ((float)(nverts2)/threadsPerBlockMean.x),1);
		
		int *h_V=new int[numBlocksMean.x];

		int * V_out_d; 
		cudaMalloc((void **)&V_out_d, sizeof(int)*numBlocksMean.x);
		
		int smemSize = threadsPerBlockMean.x*sizeof(int);
		reduceSum<<<numBlocksMean, threadsPerBlockMean, smemSize>>>(d_In_M,V_out_d,nverts2);
		// Perform final reduction in CPU
		err=cudaMemcpy(h_V, V_out_d, numBlocksMean.x*sizeof(int),cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess){
			cout << "ERROR CUDA MEM. COPY" << endl;
		} 
		long int sum = 0;
		for (int i=0; i<numBlocksMean.x; i++){
		//	cout<<" h_V[i] = "<<h_V[i]<<"\n"<<endl;
			sum += h_V[i];
		}
		float mean= 0.0f;
		mean=(float)sum/nverts2;
		cout<<"The mean of all the shortest path is: "<<mean<<"\n"<<endl;

  if (!errors){ 
    cout<<"....................................................."<<endl;
	cout<< "WELL DONE!!! No errors found ............................"<<endl;
	cout<<"....................................................."<<endl<<endl;

  }
  // Freeing memory space
  cudaFree(d_In_M);
  cudaFree(V_out_d);
  free(h_V);
  free(c_Out_M);
}

