
/* jacobi.c: Cyclic Jacobi method for finding eigenvalues and eigenvectrors
 * 
 * Author: Basileal Imana
 * Date: 06/10/16
 */

// Libriaries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include <stdbool.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

bool debug = false; // -d command line option for verbose output

/** Cuda handle error, if err is not success print error and line in code
*
* @param status CUDA Error types
*/
#define HANDLE_ERROR(status) \
{ \
   if (status != cudaSuccess) \
   { \
      fprintf(stderr, "%s failed  at line %d \nError message: %s \n", \
         __FILE__, __LINE__ ,cudaGetErrorString(status)); \
      exit(EXIT_FAILURE); \
   } \
}



// Prints a matrix to stdout
__host__ __device__ void print(float* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         printf("%.4f  ",A [i*size +j]);
      }
      printf("\n");
   }
}

// Prints a non-square matrix to stdout
__host__ __device__ void print2(float* A, int row, int col) {
   for(int i = 0; i < row; i++) {
      for(int j = 0; j < col; j++) {
         printf("%.4f  ", A[i*col+j]);
      }
      printf("\n");
   }
}

// Copies matrix elements 'from' to 'to'
void copy(float* from, float* to, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         to[i*size+j] = from[i*size+j];
      }
   }
}

// Makes matrix A an identity matrix
void eye(float* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = i; j < size; j++) {
         if(i == j) {
            A[i*size+j] = 1.0;
         } else {
            A[i*size+j] = 0.0;
            A[j*size+i] = 0.0;
         }
      }
   }
}

// Make all non diagonal elements 0
void remove_nondiag(float* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         if(i != j) {
            A[i*size+j] = 0.0;
         }
      }
   }
}

// Checks if matrix is symmetric
bool is_symmetric(float* A, int size) {
   for(int i = 0; i < size - 1; i++) {
      for(int j = i + 1; j < size; j++) {
         if(A[i*size+j] != A[j*size+i]) {
            return false;
         }
      }
   }
   return true;
}

// Calculates the square root of sum of squares of
// all off diagonal elements of A
float off(float* A, int size) {
   float sum = 0;
   for(int i = 0; i < size - 1; i++) {
      for(int j = i + 1; j < size; j++) {
         // multiply by 2 to account for other half of matrix
         sum += 2 * A[i*size+j] * A[i*size+j];
      }
   }

   return sqrt(sum);
}

// Initalizes arrays for chess tournament ordering
void chess_initialize(int* order1, int* order2, int size) {
   for(int i = 0; i <= size; i++) {
		//printf("i: %d \n", i);
      order1[i-1] = i;
      order2[i-1] = i + size;
   }
}

// Initalizes arrays for chess tournament ordering
void chess_initialize2(int* order1, int* order2, int size) {
   int curr = 0;
	for(int i = 0; i < size; i++) {
		order1[i] = ++curr;
      order2[i] = ++curr;
   }
}

// Do one permutation of chess tournament ordering
void chess_permute(int* order1, int* order2, int size) {
   // save the first element of array 2
   int temp = order2[0];
   // shift everthing in array 2 to the left
   for(int i = 0; i <= size - 2; i++) {
      order2[i] = order2[i+1];
   }
   // put last element of array 1 as last element array 2
   order2[size-1] = order1[size-1];
   // shift everything but the first two of array 1 to the right
   for(int i = size - 1; i >= 2; i--) {
      order1[i] = order1[i-1];
   }
   // put first element of array 2 as second element of array 1
   order1[1] = temp;
}

// Swaps two numbers
__host__ __device__ void swap(int* num1, int* num2) {
   int temp = *num1;
   *num1 = *num2;
   *num2 = temp;
}

// Given pivot(i,j), constructs a submatrix of rows affected J'*A
__device__ void create_sub_row(float* A, int size, int i, int j, float* A_sub) {
   for(int k = 0; k < size; k++) {
      A_sub[0 * size + k] = A[i * size + k];
      A_sub[1 * size + k] = A[j * size + k];
   }
}

// Given pivot(i,j), constructs a submatrix of row affected by A*J
__device__ void create_sub_col(float* A, int size, int i, int j, float* A_sub) {
   for(int k = 0; k < size; k++) {
      A_sub[k * 2 + 0] = A[k * size + i];
      A_sub[k * 2 + 1] = A[k * size + j];
   }
}

// Updates the original matrix's rows with changes made to submatrix
__device__ void update_sub_row(float* A, int size, int i, int j, float* A_sub) {
   for(int k = 0; k < size; k++) {
      A[i * size + k] = A_sub[0 * size + k];
      A[j * size + k] = A_sub[1 * size + k];
   }
}

// Updates the original matrix's cols with changes made to submatrix
__device__ void update_sub_col(float* A, int size, int i, int j, float* A_sub) {
   for(int k = 0; k < size; k++) {
      A[k * size + i] = A_sub[k * 2 + 0];
      A[k * size + j] = A_sub[k * 2 + 1];
   }
}

// Cacluates values of c and s for a given pivot of rotation (i,j)
__host__ __device__ void jacobi_cs(float* A, int size, int i, int j, float* c, float* s) {
   // calculate T
   float T = (A[j*size+j] - A[i*size+i]) / (2 * A[i*size+j]);

   // equation: t^2 + 2Tt - 1 = 0
   // chose the root that is smaller in absolute value
   float t;
   if(T >= 0) {
      t = -T + sqrt(1.0 + T*T);
   } else {
      t = -T - sqrt(1.0 + T*T);
   }

   // calculate c and s
   *c = 1.0 / (sqrt(1.0 + t*t));
   *s = *c * t;
}


__device__ void mul_mat(int m,int n,int k, float* a,float* b, float* c)
{
    int i,j,h;
    for(i = 0; i < m; i++)
    {
        for(j = 0; j < n; j++)
        {
             c[i * n + j] = 0;
             for(h = 0; h< k; h++)
               c[i * n + j] += + a[i * k + h] * b[h * n + j];
        }
    }
}


__global__ void jacobi_kernel1(float* D_d, float* E_d, int size, int* arr1, int* arr2, float* cc, float* ss) {
	//printf("in kernel, id = %d\n", threadIdx.x);
	// get i,j pai
	int i = arr1[threadIdx.x]-1;
	int j = arr2[threadIdx.x]-1;

	//printf("in kernel, got assigned i=%d, j=%d, id=%d\n",i,j,threadIdx.x);
	// make sure i < j
	if(i > j) {
		swap(&i, &j);
	}

	float c, s;
	c = cc[threadIdx.x];
	s = ss[threadIdx.x];
	

	// calculate values of c and s
	//jacobi_cs(D_d, size, i, j, &c, &s);
printf("in kernel, got assigned i=%d, j=%d s=%f, id=%d\n",i,j,s,threadIdx.x);
	// setup rotation matrix
	float R[] = {c, s, -s, c};
	float R_t[] = {c, -s, s, c};


	//printf("in kernel allocate memory, id = %d\n", threadIdx.x);

	// Submatrices (2xn or nx2 size) for storing intermediate results
   float* D_sub = (float *) malloc(sizeof(float) * 2 * size);
   float* E_sub = (float *) malloc(sizeof(float) * 2 * size);
   float* X_sub = (float *) malloc(sizeof(float) * 2 * size);

	// get submatrix of rows of D that will be affected by R' * D
   create_sub_row(D_d, size, i, j, D_sub);

   // sgemm calculates C = alpha*A*B + beta*C
   float alp = 1.0;
   float bet = 0.0;
	const float* alpha = &alp;
	const float* beta = &bet;
	
	//printf("in kernel creating handle, id = %d\n", threadIdx.x);

	// Create a handle for CUBLAS
	//cublasHandle_t handle;
	//cublasCreate(&handle);

	//printf("in kernel calling cublas sgemm id = %d\n", threadIdx.x);


   // calculate X_sub = R' * D_sub
   //cublasSgemm(handle,CUBLAS_OP_T, CUBLAS_OP_N, \
               2, size, 2, alpha, R, 2, D_sub, size, beta, X_sub, size);
//__syncthreads();
/*if(threadIdx.x == 0) {
	printf("before: \n");
   print(D_d,size); printf("\n");
   print2(D_sub,2,size);
}*/
 
	mul_mat(2,size,2,R_t,D_sub,X_sub);

	//printf("in kernel done with cublas id = %d \n", threadIdx.x);
	// Destroy CUBLAS handle
	//cublasDestroy(handle);

   // update D
   update_sub_row(D_d,size,i,j,X_sub);
//__syncthreads();
/*if(threadIdx.x == 0) {
	printf("after \n");
   print2(X_sub,  2,size); printf("\n");
   print(D_d,size);printf("\n");
}*/

/* __________________________________*/
__syncthreads();
/*  // get submatrix of cols of D that will be affected by D * R
   create_sub_col(D_d,size,i,j,D_sub);


   // create a handle for CUBLAS
   //cublasHandle_t handle;
   //cublasCreate(&handle);

   // calculate X_sub = D_sub * R
   //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, \
               size, 2, 2, alpha, D_sub, size, R, 2, beta, X_sub, size);
//   if(threadIdx.x == 0) {
   //printf("before: \n");
 //  print(D_d,size); printf("\n");
//   print2(D_sub,size,2);
//}


    mul_mat(size,2,2,D_sub,R,X_sub);
   // update D
   update_sub_col(D_d,size,i,j,X_sub);


   // get submatrix of cols of E that iwll be affected by E * R
   create_sub_col(E_d,size,i,j,E_sub);

   // calculate X_sub = E_sub * R
   //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, \
                    size, 2, 2, alpha, E_sub, size, R, 2, beta, X_sub, size);

   mul_mat(size,2,2,E_sub,R,X_sub);

   //cublasDestroy(handle);

   // update E
   update_sub_col(E_d,size,i,j,X_sub);*/
/*______________________________________*/

	//printf("in kenel about to free memory id = %d \n", threadIdx.x);
	// free memory
	free(D_sub);
	free(E_sub);
	free(X_sub);
	//printf("in kernel thread done id = %d \n", threadIdx.x);
}


__global__ void jacobi_kernel2(float* D_d, float* E_d, int size, int* arr1, int* arr2, float* cc, float* ss) {
	
	// get i,j pair
	int i = arr1[threadIdx.x] - 1;
	int j = arr2[threadIdx.x] - 1;

	 //printf("in kernel2, got assigned i=%d, j=%d, id=%d\n",i,j,threadIdx.x);


	// make sure i < j
	if(i > j) {
		swap(&i,&j);
	}



	float c, s;
	   c = cc[threadIdx.x];
   s = ss[threadIdx.x];


   // calculate values of c and s
   //jacobi_cs(D_d, size, i, j, &c, &s);

	printf("in kernel2, got assigned i=%d, j=%d s=%f, id=%d\n",i,j,s,threadIdx.x);

   // setup rotation matrix
   float R[] = {c, s, -s, c};

   // Submatrices (2xn or nx2 size) for storing intermediate results
   float* D_sub = (float *) malloc(sizeof(float) * 2 * size);
   float* E_sub = (float *) malloc(sizeof(float) * 2 * size);
   float* X_sub = (float *) malloc(sizeof(float) * 2 * size);

	// get submatrix of cols of D that will be affected by D * R
   create_sub_col(D_d,size,i,j,D_sub);

	// sgemm calculate C = alpha*A*B + beta*C
	float alp = 1.0;
	float bet = 0.0;
   const float* alpha = &alp;
   const float* beta = &bet;


   // create a handle for CUBLAS
	//cublasHandle_t handle;
	//cublasCreate(&handle);

	// calculate X_sub = D_sub * R
   //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, \
               size, 2, 2, alpha, D_sub, size, R, 2, beta, X_sub, size);
/*   if(threadIdx.x == 0) {
   printf("before: \n");
   print(D_d,size); printf("\n");
   print2(D_sub,size,2);
}*/


	 mul_mat(size,2,2,D_sub,R,X_sub);
   // update D
   update_sub_col(D_d,size,i,j,X_sub);
//__syncthreads();
//if(threadIdx.x == 0) {
//   printf("after2 \n");
//   print2(X_sub,size,2); printf("\n");
//   print(D_d,size);printf("\n");
//}

   // get submatrix of cols of E that iwll be affected by E * R
   create_sub_col(E_d,size,i,j,E_sub);

   // calculate X_sub = E_sub * R
   //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, \
                    size, 2, 2, alpha, E_sub, size, R, 2, beta, X_sub, size);

	mul_mat(size,2,2,E_sub,R,X_sub);

	//cublasDestroy(handle);

   // update E
   update_sub_col(E_d,size,i,j,X_sub);

	// free memory
	free(D_sub);
	free(E_sub);
	free(X_sub);
	//printf("in kernel2 thread done id = %d \n", threadIdx.x);

}

// Jacobi method
void jacobi(float* A, float* D, float* E, int size, float epsilon) {
   // initialize D and E
   copy(A, D, size);
   eye(E, size);

   // device memory pointers
   float *D_d, *E_d;

	// chess tournament ordering, and corresponding values of c and s pointers
	int *arr1, *arr2;
	float *cc, *ss;
	//int *arr1_d, *arr2_d;
	cudaError_t cudaStatus;


	printf("about to allocate unified memory\n");
	// allocate unified memory
	cudaMallocManaged(&arr1, sizeof(int) * size/2);
	cudaMallocManaged(&arr2, sizeof(int) * size/2);

	// arrray to calculate and store corresponding values of c and s
	cudaMallocManaged(&cc, sizeof(float) * size/2);
	cudaMallocManaged(&ss, sizeof(float) * size/2);
	
	//arr1 = (int* ) malloc(sizeof(int) * size/2);
	//arr2 = (int* ) malloc(sizeof(int) * size/2);

	printf("about to allocate device memory\n");
   // allocate device memory
   cudaMallocManaged(&D_d, sizeof(float) * size*size);
   cudaMalloc((void **) &E_d, sizeof(float) * size*size);

	//cudaMalloc((void **) &arr1_d, sizeof(int) * size/2);
	//cudaMalloc((void **) &arr2_d, sizeof(int) * size/2);

	printf("about to copy to device\n");

	copy(D,D_d,size);
   // copy matrices to device
   //cudaMemcpy(D_d, D, sizeof(float) * size*size, cudaMemcpyHostToDevice);  
   cudaMemcpy(E_d, E, sizeof(float) * size*size, cudaMemcpyHostToDevice);
	
	printf("about to launch kernels\n");
	while(off(D_d,size) > epsilon) {
		// initialize ordering of i,j pairs
		chess_initialize2(arr1, arr2, size/2);
		
		for(int h = 0; h < size-1; h++) {
			for(int k = 0; k < size/2; k++) {
				int i = arr1[k] - 1;
				int j = arr2[k] - 1;
				if(i > j) swap(&i,&j);
				jacobi_cs(D_d, size, i, j,&cc[k],&ss[k]);
			}
			//printf("Launch: %d\n",i);
			// launch kernel 1 size/2 threads 
			jacobi_kernel1<<<1,size/2>>>(D_d, E_d, size, arr1, arr2, cc,ss);
		
			// synchronize
			cudaStatus = cudaDeviceSynchronize();	
			HANDLE_ERROR(cudaStatus);

			jacobi_kernel2<<<1,size/2>>>(D_d, E_d, size, arr1, arr2, cc, ss);
			// synchronize
         cudaStatus = cudaDeviceSynchronize();
         HANDLE_ERROR(cudaStatus);

			// do next permutation of i, j pairs
			chess_permute(arr1, arr2, size/2);
		}

      /*// initialize ordering of i,j pairs
      chess_initialize2(arr1, arr2, size/2);

      for(int i = 0; i < size-1; i++) {
         //printf("Launch2: %d\n",i);
         // launch kernel 1 size/2 threads
         jacobi_kernel2<<<1,size/2>>>(D_d, E_d, size, arr1, arr2);

         // synchronize
         cudaStatus = cudaThreadSynchronize();
			HANDLE_ERROR(cudaStatus);

         // do next permutation of i, j pairs
         chess_permute(arr1, arr2, size/2);
      }*/
	
		print(D_d,size);
		printf("\n");
		break;		

   }
	
	copy(D_d,D,size);
	cudaMemcpy(E, E_d, sizeof(float) * size*size, cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(arr1);
	cudaFree(arr2);
	//cudaFree(arr1_d);
	//cudaFree(arr2_d);
	//cudaFiree(arr1);
	//cudaFree(arr2);
	cudaFree(D_d);
	cudaFree(E_d);
}

// Main
int main(int argc, char** argv) {

   // process command line arguments
   int r;
   while ((r = getopt(argc, argv, "d")) != -1) {
      switch(r)
      {
         case 'd':
            debug = true;
            break;
         default:
            exit(1);
      }
   }

   // read matrix size from stdin
   int size;
   scanf("%d",&size);

   // initialize array
   float* A = (float*) malloc(sizeof(float) * size * size);
   float* D = (float*) malloc(sizeof(float) * size * size);
   float* E = (float*) malloc(sizeof(float) * size * size);

   // read matrix from stdin
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         scanf("%f", &A[i * size + j]);
      }
   }

   // make sure matrix is symmetric
   if(!is_symmetric(A, size)) {
      printf("Error: Given matrix not symmetric!\n");
      return 0;
   }
   
   if(debug) {
      printf("Input matrix A: \n");
      print(A, size);
      printf("\n");
   }

   // desired accuracy
   float epsilon = 0.01;

   // call facobi method
   jacobi(A, D, E, size, epsilon);
   remove_nondiag(D, size);

   // output results
   printf("\n");
   printf("______Results______\n");
   printf("Eigenvalues on the diagonal:\n");
   print(D, size);
   printf("\n");
   printf("Corresponding eigenvectors:\n");
   print(E, size);
   printf("\n");

	// clean up
	free(A);
	free(D);
	free(E);

   return 0;
}

