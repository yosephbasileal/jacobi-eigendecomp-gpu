/* Utility functions for CUDA
 *
 * Author: Basileal Imana
 * Date: 07/04/16
 */

// Libriaries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/complex.h>
#include <stdbool.h>

// Macro for creating complex nubmers
#define create_comp thrust::complex<double>
typedef thrust::complex<double> comp;

// Prints a square matrix to stdout
__host__ __device__ void print(comp* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         printf("%+.4f%+.4fi ", A[i*size+j].real(), A[i*size+j].imag());
      }
      printf("\n");
   }
}

// Prints a square matrix to stdout, column major format
__host__ __device__ void print_cm(comp* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         printf("%+.4f%+.4fi ", A[j*size+i].real(), A[j*size+i].imag());
      }
      printf("\n");
   }
}

// Prints a non-square matrix to stdout
__host__ __device__ void print2(comp* A, int row, int col) {
   for(int i = 0; i < row; i++) {
      for(int j = 0; j < col; j++) {
         printf("%+.15f%+.15fi ", A[i*col+j].real(), A[i*col+j].imag());
      }
      printf("\n");
   }
}

// Prints a non-square matrix - output is valid matlab matrix form
__host__ __device__ void print3(comp* A, int row, int col) {
   printf("M = [");
   for(int i = 0; i < row; i++) {
      for(int j = 0; j < col; j++) {
         printf("%+.15f%+.15fi ", A[i*col+j].real(), A[i*col+j].imag());
      }
      if(i != row-1) {
         printf(";\n");
      }
   }
   printf("];\n");
}

// Prints value of a complex number
__host__ __device__ void printd(comp val, const char* name) {
   printf("%s: %+.15f%+.15fi\n", name, val.real(), val.imag());
}

// Comparator for sorting an array of flaots using qsort
__host__ __device__ int compare(const void* a, const void* b) {
   comp f_a = *((comp *) a );
   comp f_b = *((comp *) b );

   if(abs(f_a) == abs(f_b) ) return 0;
   else if ( abs(f_a) < abs(f_b) ) return -1;
   else return 1;
}

// Copies square matrix elements 'from' to 'to'
__host__ __device__ void copy(comp* from, comp* to, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         to[i*size+j] = from[i*size+j];
      }
   }
}

// Makes matrix A an identity matrix
__host__ __device__ void eye(comp* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = i; j < size; j++) {
         if(i == j) {
            A[i*size+j] = create_comp(1.0,0.0);
         } else {
            A[i*size+j] = create_comp(0.0,0.0);
            A[j*size+i] = create_comp(0.0,0.0);;
         }
      }
   }
}

// Make all non diagonal elements of square matrix A 0
__host__ __device__ void remove_nondiag(comp* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         if(i != j) {
            A[i*size+j] = create_comp(0.0,0.0);
         }
      }
   }
}

// Calculates the square root of sum of squares of
// all off diagonal elements of symmetric matrix A
__host__ __device__ double off(comp* A, int size) {
   double sum = 0;
   for(int i = 0; i < size - 1; i++) {
      for(int j = i + 1; j < size; j++) {
         // multiply by 2 to account for other half of matrix
         sum += 2 * pow(abs(A[i*size+j]),2);
      }
   }

   return sqrt(sum);
}

// Calculates norm of strictly lower triangluar part
// of matrix A
__host__ __device__ double lower(comp * A, int size) {
   double sum = 0;
   for(int i = 0; i < size; i++) {
      for(int j = 0; j <= i - 1; j++) {
         sum += pow(abs(A[i*size+j]),2);
      }
   }

   return sqrt(sum);
}

// Get a vecotr that contains diagonal elements of matrix A
__host__ __device__ void get_diagonals(comp* d, comp* A, int size) {
   for(int i = 0; i < size; i++) {
      d[i] = A[i*size+i];
   }
}

// Get ith row of a square matrix
__host__ __device__ void get_ith_row(comp* A, comp* row, int size, int i) {
   for(int j = 0; j < size; j++) {
      row[j] = A[i*size+j];
   }
}

// Get jth col of a square matrix
__host__ __device__ void get_jth_col(comp* A, comp* col, int size, int j) {
   for(int i = 0; i < size; i++) {
      col[i] = A[i*size+j];
   }
}

// Create a random square complex matrix
 void create_mat(comp* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         double a = -1 + 2*((double)rand())/RAND_MAX; //rand between -1 and 1
         double b = -1 + 2*((double)rand())/RAND_MAX;
         A[i*size+j] = create_comp(a,b);
      }
   }
}

// Multiplies A(mxk) matrix by B(kxn) matrix
__host__ __device__ void mul_mat(int m,int n,int k, comp* a, comp* b, comp* c) {
   int i,j,h;
   for(i = 0; i < m; i++) {
      for(j = 0; j < n; j++) {
         c[i * n + j] = create_comp(0.0,0.0);
         for(h = 0; h < k; h++) {
            c[i * n + j] += a[i * k + h] * b[h * n + j];
         }
      }
   }
}

// Subtracts matrix b from a, result in c
__host__ __device__ void sub_mat(comp* a, comp* b, comp* c, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         c[i*size+j] = a[i*size+j] - b[i*size+j];
      }
   }
}

// Calculates euclidean norm of a square matrix
__host__ __device__ double norm_mat(comp* a, int size) {
   double sum = 0;
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         sum += pow(abs(a[i*size+j]),2);
      }
   }
   return sqrt(sum);
}

// Calculate residual error of eigendecomposition
__host__ __device__ double residual(comp* A, comp* P, comp* D, int size) {
   comp* AP = (comp *) malloc(sizeof(comp) * size*size);
   comp* PD = (comp *) malloc(sizeof(comp) * size*size);
   comp* DIFF = (comp *) malloc(sizeof(comp) * size*size);

   mul_mat(size,size,size,A,P,AP);
   mul_mat(size,size,size,P,D,PD);

   sub_mat(AP,PD,DIFF,size);

   return norm_mat(DIFF,size);
}

// Converts from column major to row major (transposes a matrix)
__host__ __device__ void cm_to_rm(comp* A, int size) {
   comp* B = (comp *) malloc(sizeof(comp) * size*size);
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         B[i*size+j] = A[j*size+i];
      }
   }

   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         A[i*size+j] = B[i*size+j];
      }
   }
}
