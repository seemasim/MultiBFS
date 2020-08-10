#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <fstream>
#include <algorithm>
#include <string>
#include  <thrust/equal.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>



void printMatrix(int m, int n, const float*A, int lda, const char* name)
{
	for(int row = 0 ; row < m ; row++){
		for(int col = 0 ; col < n ; col++){
			float Areg = A[row + col*lda];
			printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
		}
	}
}


struct my_policy : thrust::device_execution_policy<my_policy> {};

int main(int argc, char*argv[])
{

	cusparseMatDescr_t descrA = NULL;
	cusparseMatDescr_t descrB = NULL;
	cusparseMatDescr_t descrC = NULL;

	cublasStatus_t cublasStat = CUBLAS_STATUS_SUCCESS;
	cusparseStatus_t cusparseStat = CUSPARSE_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	
	const int n = 4847571;
	const size_t nnzA = 68993773 + n;
	//const int n = 7;
	//const size_t nnzA = 19;
	size_t nnzB = n;
	
	const int *csrRowPtrA = (const int*)malloc(sizeof(const int)*(n+1));
	const int *csrColIndA = (const int*)malloc(sizeof(const int)*(nnzA));
	const float csrValA[nnzA] = { 0.0 };


	int *csrRowPtrB = (int *)malloc(sizeof(int)*(n+1));
	int *csrColIndB = (int *)malloc(sizeof(int)*(nnzB));
	float csrValB[nnzB];

	int i;
	for(i=0;i<n;i++){
		*(csrColIndB + i) = i;
		*(csrRowPtrB + i) = i;
		csrValB[i] = 1.0;
	}
	*(csrRowPtrB + i) = i;


	{
		//std::ifstream file("value_pokec.txt");
		std::ifstream file("value.txt");
		std::string str;
		int i = 0;
		while (std::getline(file, str)) {
			float &ptr = const_cast <float &>(csrValA[i]); 
			ptr = (const float)atoi(str.c_str()); 
			i++;
		}
	}

	{
		//std::ifstream file("indices_pokec.txt");
		std::ifstream file("indices.txt");
		int i = 0;
		std::string str;
		while (std::getline(file, str)) {
			int &ptr = const_cast <int &>(csrColIndA[i]);
			ptr = (const int)atoi(str.c_str());
			i++;
		}
	}

	{
		//std::ifstream file("indptr_pokec.txt");
		std::ifstream file("indptr.txt");
		int i = 0;
		std::string str;
		while (std::getline(file, str)) {
			int &ptr = const_cast <int &>(csrRowPtrA[i]);
			ptr = (const int)atoi(str.c_str());
			i++;
		}
	}



	int cscColPtrA[n+1];
	int cscRowIndA[nnzA];
	float cscValA[nnzA];

	int *d_csrRowPtrA = NULL;
	int *d_csrColIndA = NULL;
	float *d_csrValA = NULL;

	int *d_csrRowPtrB = NULL;
	int *d_csrColIndB = NULL;
	float *d_csrValB = NULL;

	int *d_cscColPtrA = NULL;
	int *d_cscRowIndA = NULL;
	float *d_cscValA = NULL;


	/* step 2: configuration of matrix A */
	cusparseStat = cusparseCreateMatDescr(&descrA);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

	cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

	/* configuration of matrix B */

	cusparseStat = cusparseCreateMatDescr(&descrB);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);   

	cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL );

	/* step 3: copy A,B and x0 to device */
	cudaStat1 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (n+1) );
	cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA );
	cudaStat3 = cudaMalloc ((void**)&d_csrValA   , sizeof(float) * nnzA );

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	cudaStat1 = cudaMalloc ((void**)&d_cscColPtrA, sizeof(int) * (n+1) );
	cudaStat2 = cudaMalloc ((void**)&d_cscRowIndA, sizeof(int) * nnzA );
	cudaStat3 = cudaMalloc ((void**)&d_cscValA   , sizeof(float) * nnzA );

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	cudaStat1 = cudaMalloc ((void**)&d_csrRowPtrB, sizeof(int) * (n+1) );
	thrust::device_ptr<int> v2RowPtrB(d_csrRowPtrB);
	cudaStat2 = cudaMalloc ((void**)&d_csrColIndB, sizeof(int) * n );
	thrust::device_ptr<int> v1ColIndB(d_csrColIndB);
	cudaStat3 = cudaMalloc ((void**)&d_csrValB   , sizeof(float) * n );

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);


	cudaStat1 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (n+1)   , cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA    , cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(d_csrValA   , csrValA   , sizeof(float) * nnzA , cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	cudaStat1 = cudaMemcpy(d_csrRowPtrB, csrRowPtrB, sizeof(int) * (n+1)   , cudaMemcpyHostToDevice);

	cudaStat2 = cudaMemcpy(d_csrColIndB, csrColIndB, sizeof(int) * n    , cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(d_csrValB   , csrValB   , sizeof(float) * n , cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	cublasHandle_t cublasH = NULL;
	cusparseHandle_t cusparseH = NULL;
	cudaStream_t stream = NULL;

	/* step 1: create cublas/cusparse handle, bind a stream */
	cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == cudaStat1);

	cublasStat = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublasStat);

	cublasStat = cublasSetStream(cublasH, stream);
	assert(CUBLAS_STATUS_SUCCESS == cublasStat);

	cusparseStat = cusparseCreate(&cusparseH);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

	cusparseStat = cusparseSetStream(cusparseH, stream);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

	cusparseStat = cusparseScsr2csc(cusparseH, 
			n, 
			n, 
			nnzA,
			d_csrValA, 
			d_csrRowPtrA,
			d_csrColIndA, 
			d_cscValA, 
			d_cscRowIndA,
			d_cscColPtrA, 
			CUSPARSE_ACTION_NUMERIC, 
			CUSPARSE_INDEX_BASE_ZERO);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);


	cudaStat1 = cudaMemcpy(cscValA, d_cscValA, sizeof(float) * nnzA, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(cscRowIndA, d_cscRowIndA, sizeof(int) * nnzA, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(cscColPtrA, d_cscColPtrA, sizeof(int) * (n+1), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	 

	delete[] csrRowPtrA;
	delete [] csrColIndA;
	

	//Configure matrix C
	cusparseStat = cusparseCreateMatDescr(&descrC);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);   

	cusparseStat = cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);   

	cusparseStat = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL );
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);   

	int iteration = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float time_in_ms = 0;

	while(1){ 
		printf("iteration==%d\n",iteration );

		int *d_csrRowPtrC = NULL;
		int *d_csrColIndC = NULL;
		float *d_csrValC = NULL; //x0

		
		int baseC,nnzC;

		// nnzTotalDevHostPtr points to host memory
		int *nnzTotalDevHostPtr = (int*)&nnzC;

		cusparseStat = cusparseSetPointerMode(cusparseH, CUSPARSE_POINTER_MODE_HOST);
		assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);


		cudaStat1 = cudaMalloc((void**)&d_csrRowPtrC, sizeof(int)*(n+1));
		thrust::device_ptr<int> v2RowPtrC(d_csrRowPtrC);
	
		assert(cudaSuccess == cudaStat1);
			

		//d_cscColPtrA is used because we want the transpose of matrix A to be used
		float time_nnzC;
		cudaEventRecord(start,stream);
		cusparseStat = cusparseXcsrgemmNnz(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, n,
				descrA, nnzA, d_cscColPtrA, d_cscRowIndA,
				descrB, nnzB, d_csrRowPtrB, d_csrColIndB,
				descrC, d_csrRowPtrC, nnzTotalDevHostPtr);


		assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);  
		cudaEventRecord(stop,stream);

		if (NULL != nnzTotalDevHostPtr){
			nnzC = *nnzTotalDevHostPtr;
		}else{
			cudaMemcpy(&nnzC, d_csrRowPtrC+n, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&baseC, d_csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
			nnzC -= baseC;
		}
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_nnzC, start, stop);

		cudaMalloc((void**)&d_csrColIndC, sizeof(int)*nnzC);
		thrust::device_ptr<int> v1ColIndC(d_csrColIndC);
		cudaMalloc((void**)&d_csrValC, sizeof(float)*nnzC);

		printf("nnzC=%d\n",nnzC);

	
		float time_mm;
		cudaEventRecord(start, stream);
		cusparseStat = cusparseScsrgemm(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, n,
				descrA, nnzA,
				d_cscValA, d_cscColPtrA, d_cscRowIndA,
				descrB, nnzB,
				d_csrValB, d_csrRowPtrB, d_csrColIndB,
				descrC,
				d_csrValC, d_csrRowPtrC, d_csrColIndC);
		assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
		cudaEventRecord(stop, stream);

		

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_mm, start, stop);

		

		thrust::device_ptr<int> v2RowPtrB(d_csrRowPtrB);
		thrust::device_ptr<int> v1ColIndB(d_csrColIndB);
		bool flag1 = false;
		bool flag2 = false;
		//size_t N = nnzB;
		my_policy exec;

		//compare matrix B with matrix C because matrix B had previous A*B and C has new A*B
		float time_eqlComparison;
		if(iteration != 0){			
			
			cudaEventRecord(start, stream);printf("recording...\n");
			flag1 = thrust::equal(exec, v1ColIndB, v1ColIndB + nnzB, v1ColIndC);			
			flag2 = thrust::equal(exec, v2RowPtrB, v2RowPtrB + n, v2RowPtrC);
			cudaEventRecord(stop, stream);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time_eqlComparison, start, stop);
			if(flag1 == true && flag2 == true){
				printf("**CONVERGED** the previous two iteration are same\n");
				if (d_csrRowPtrC  ) cudaFree(d_csrRowPtrC);
				if (d_csrColIndC  ) cudaFree(d_csrColIndC);
				if (d_csrValC     ) cudaFree(d_csrValC);
				if (descrC        ) cusparseDestroyMatDescr(descrC);
				break;
			}
		}
		nnzB = nnzC;


		if (d_csrRowPtrB  ) cudaFree(d_csrRowPtrB);
		if (d_csrColIndB  ) cudaFree(d_csrColIndB);
		if (d_csrValB     ) cudaFree(d_csrValB);


		cudaStat1 = cudaMalloc ((void**)&d_csrRowPtrB, sizeof(int) * (n+1) );		
		cudaStat2 = cudaMalloc ((void**)&d_csrColIndB, sizeof(int) * nnzB );		
		cudaStat3 = cudaMalloc ((void**)&d_csrValB   , sizeof(float) * nnzB );
		

		assert(cudaSuccess == cudaStat1);
		assert(cudaSuccess == cudaStat2);
		assert(cudaSuccess == cudaStat3);


		cudaStat1 = cudaMemcpy(d_csrRowPtrB, d_csrRowPtrC, sizeof(int) * (n+1), cudaMemcpyDeviceToDevice);
		cudaStat2 = cudaMemcpy(d_csrColIndB, d_csrColIndC, sizeof(int) * nnzB    , cudaMemcpyDeviceToDevice);
		cudaStat3 = cudaMemcpy(d_csrValB   , d_csrValC   , sizeof(float) * nnzB , cudaMemcpyDeviceToDevice);
		assert(cudaSuccess == cudaStat1);
		assert(cudaSuccess == cudaStat2);
		assert(cudaSuccess == cudaStat3);

		if (d_csrRowPtrC  ) cudaFree(d_csrRowPtrC);
		if (d_csrColIndC  ) cudaFree(d_csrColIndC);
		if (d_csrValC     ) cudaFree(d_csrValC);

		iteration++;
		float x=time_nnzC + time_mm ;
		printf("time elapsed in this iteration=%f\n",x);
		time_in_ms += x;
		
		} //end while

	printf("total time==%f\n",time_in_ms);
		//if iteration is 1 then store the result matrix to X for the 
		//remaining iterations compare the two matrices.
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		if (cublasH       ) cublasDestroy(cublasH);
		if (cusparseH     ) cusparseDestroy(cusparseH);
		if (stream        ) cudaStreamDestroy(stream);
		/* free resources */
		if (d_csrRowPtrA  ) cudaFree(d_csrRowPtrA);
		if (d_csrColIndA  ) cudaFree(d_csrColIndA);
		if (d_csrValA     ) cudaFree(d_csrValA);
		if (descrA        ) cusparseDestroyMatDescr(descrA);

		if (d_csrRowPtrB  ) cudaFree(d_csrRowPtrB);
		if (d_csrColIndB  ) cudaFree(d_csrColIndB);
		if (d_csrValB     ) cudaFree(d_csrValB);
		if (descrB        ) cusparseDestroyMatDescr(descrB);


		printf("9\n");
		cudaDeviceReset();

		return 0;
	}
