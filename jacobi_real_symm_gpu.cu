
/* Cyclic Jacobi method for finding eigenvalues and eigenvectrors
 * of real symmetric matrices
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
#include <time.h>
#include "utils.h"

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

bool debug = false; // -d command line option for verbose output
bool output = false; // -p command line option for ouputting results

double epsilon = 0.01; // -e command line option for desired accuracy
int num_sweeps = 6; // -s command line option for number of sweeps

// Initalizes arrays for chess tournament ordering
void chess_initialize(int* order1, int* order2, int size) {
   int curr = -1;
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

// Cacluates values of c and s for a given pivot of rotation (i,j)
void jacobi_cs(double* A, int size, int i, int j, double* c, double* s) {
   // calculate T
   double T = (A[j*size+j] - A[i*size+i]) / (2 * A[i*size+j]);

   // equation: t^2 + 2Tt - 1 = 0
   // chose the root that is smaller in absolute value
   double t;
   if(T >= 0) {
      t = -T + sqrt(1.0 + T*T);
   } else {
      t = -T - sqrt(1.0 + T*T);
   }

   // calculate c and s
   *c = 1.0 / (sqrt(1.0 + t*t));
   *s = *c * t;
}

__global__ void jacobi_kernel1(double* D, double* X, int size, int* arr1, int* arr2, double* cc, double* ss) {

   int tid = threadIdx.x;

   // get i,j pair, all threads in block operate on row i and row j
   int i = arr1[blockIdx.x];
   int j = arr2[blockIdx.x];

   // make sure i < j
   if(i > j) {
      int temp = i;
      i = j;
      j = temp;
   }

   // get precaculated values of c and s for current values of i and j
   double c = cc[blockIdx.x];
   double s = ss[blockIdx.x];

   // setup rotation matrix
   double R_T[] = {c, -s, s, c};

   // get row i and row j elements for current thread
   double row_i = D[i*size+tid];
   double row_j = D[j*size+tid];

   // calculate X = R' * D, X is column major array
   X[tid*size+i] = R_T[0] * row_i + R_T[1] * row_j;
   X[tid*size+j] = R_T[2] * row_i + R_T[3] * row_j;
}


__global__ void jacobi_kernel2(double* D, double* E, double* X, int size, int* arr1, int* arr2, double* cc, double* ss) {

   int tid = threadIdx.x;

   // get i,j pair, all threads in block operate on col i and col j
   int i = arr1[blockIdx.x];
   int j = arr2[blockIdx.x];

   // make sure i < j
   if(i > j) {
      int temp = i;
      i = j;
      j = temp;
   }

   // get precaculated values of c and s for current values of i and j
   double c = cc[blockIdx.x];
   double s = ss[blockIdx.x];

   // setup rotation matrix
   double R[] = {c, s, -s, c};

   // get col i and col j elements of X for current thread
   double x_col_i = X[i*size+tid];
   double x_col_j = X[j*size+tid];

   // calculate D = X * R, X is column major array
   D[i*size+tid] = x_col_i * R[0] + x_col_j * R[2];
   D[j*size+tid] = x_col_i * R[1] + x_col_j * R[3];

   // get col i and col j elements of E for current thread
   double e_col_i = E[i*size+tid];
   double e_col_j = E[j*size+tid];

   // caclulate E = E * R, E is column major array
   E[i*size+tid] = e_col_i * R[0] + e_col_j * R[2];
   E[j*size+tid] = e_col_i * R[1] + e_col_j * R[3];
}

// Jacobi method
void jacobi(double* A, double* D, double* E, int size, double epsilon, int num_sweeps) {
   printf("Initializing jacobi matrices...\n");

   // initialize D and E
   copy(A, D, size);
   eye(E, size);

   // device memory pointers for matrices
   double *D_d, *E_d, *X_d; //E and X are column major arrays

   // chess tournament ordering arr1 stores i, arr2 stroes j
   int *arr1, *arr2;

   // store c and s values for corresponding (i,j) pair
   double *cc, *ss;

   cudaError_t cudaStatus;

   // allocate unified memory
   cudaMallocManaged(&arr1, sizeof(int) * size/2);
   cudaMallocManaged(&arr2, sizeof(int) * size/2);
   cudaMallocManaged(&cc, sizeof(double) * size/2);
   cudaMallocManaged(&ss, sizeof(double) * size/2);
   cudaMallocManaged(&D_d, sizeof(double) * size*size);

   // allocate device memory
   cudaMalloc((void **) &E_d, sizeof(double) * size*size);
   cudaMalloc((void **) &X_d, sizeof(double) * size*size);

   // copy matrices to device
   copy(D,D_d,size);
   cudaMemcpy(E_d, E, sizeof(double) * size*size, cudaMemcpyHostToDevice);

   int sweep_count = 0;
   double offA;

   // do sweeps
   while((offA = off(D_d,size)) > epsilon && (sweep_count < num_sweeps)) {

      sweep_count++;
      printf("Doing sweep #%d  off(D) = %.8lf \n", sweep_count, offA);

      // initialize ordering of i,j pairs
      chess_initialize(arr1, arr2, size/2);

      for(int h = 0; h < size-1; h++) {

         // precalcuate values of c and s for current permuationt so
         // that both kernels use the same rotation matrix
         for(int k = 0; k < size/2; k++) {
            int i = arr1[k];
            int j = arr2[k];
            if(i > j) {
               int temp = i;
               i = j;
               j = temp;
                                }
            jacobi_cs(D_d, size, i, j,&cc[k],&ss[k]);
         }

         // launch kernel 1
         jacobi_kernel1<<<size/2,size>>>(D_d, X_d, size, arr1, arr2, cc,ss);

         // synchronize
         cudaStatus = cudaDeviceSynchronize();
         HANDLE_ERROR(cudaStatus);

         // launch kernel 2
         jacobi_kernel2<<<size/2,size>>>(D_d, E_d, X_d, size, arr1, arr2, cc, ss);

         // synchronize
         cudaStatus = cudaDeviceSynchronize();
         HANDLE_ERROR(cudaStatus);

         // do next permutation of i, j pairs
         chess_permute(arr1, arr2, size/2);
      }

      if(debug) {
         printf("One sweep done. New matrix D: \n");
         print(D_d, size);
         printf("\n");
      }
   }

   // copy to host
   copy(D_d,D,size);
   cudaMemcpy(E, E_d, sizeof(double) * size*size, cudaMemcpyDeviceToHost);

   // free memory
   cudaFree(arr1);
   cudaFree(arr2);
   cudaFree(cc);
   cudaFree(ss);
   cudaFree(D_d);
   cudaFree(E_d);
   cudaFree(X_d);
}

// Main
int main(int argc, char** argv) {

   // process command line arguments
   int r;
   while ((r = getopt(argc, argv, "dps:e:")) != -1) {
      switch(r)
      {
         case 'd':
            debug = true;
            break;
         case 'p':
            output = true;
            break;
         case 's':
            num_sweeps = atoi(optarg);
            break;
         case 'e':
            epsilon = atof(optarg);
            break;
         default:
            exit(1);
      }
   }

   printf("Reading matrix from file...\n");
   // read matrix size from stdin
   int size;
   scanf("%d",&size);

   // initialize arrays
   double* A = (double*) malloc(sizeof(double) * size * size);
   double* D = (double*) malloc(sizeof(double) * size * size);
   double* E = (double*) malloc(sizeof(double) * size * size);

   // array to store eigenvalues
   double* ei = (double *) malloc(sizeof(double) * size);

   // read matrix from stdin
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         scanf("%lf", &A[i * size + j]);
      }
   }

   // make sure matrix is symmetric
   if(!is_symmetric(A, size)) {
      printf("Warning: Given matrix not symmetric!\n");
      //return 0;
   }

   if(debug) {
      printf("Input matrix A: \n");
      print(A, size);
      printf("\n");
   }

   clock_t begin, end;
   double time_spent;

   begin = clock();

   // call facobi method
   jacobi(A, D, E, size, epsilon, num_sweeps);

   end = clock();
   time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

   printf("Post-processing...\n");
   remove_nondiag(D, size);
   get_diagonals(ei, D, size);
   qsort(ei, size, sizeof(double), compare);

   // output results
   if(output) {
      printf("\n");
      printf("Sorted Eigenvalues:\n");
      for(int i = 0; i < size; i++) {
         printf("%.15lf\n",ei[i]);
      }
      printf("\n");
      //printf("Eigenvectors:\n");
      //print_cm(E, size);
      //printf("\n");
   }

   printf("Execution time of Jacobi: %lf\n", time_spent);

   // clean up
   free(A);
   free(D);
   free(E);

   return 0;
}

