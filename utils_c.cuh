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
void print(comp* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         printf("%+.4f%+.4fi ", A[i*size+j].real(), A[i*size+j].imag());
      }
      printf("\n");
   }
}

// Prints a square matrix to stdout, column major format
void print_cm(comp* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         printf("+%.4f%+.4fi ", A[j*size+i].real(), A[j*size+i].imag());
      }
      printf("\n");
   }
}

// Prints a non-square matrix to stdout
void print2(comp* A, int row, int col) {
   for(int i = 0; i < row; i++) {
      for(int j = 0; j < col; j++) {
         printf("%+.4f%+.4fi ", A[i*col+j].real(), A[i*col+j].imag());
      }
      printf("\n");
   }
}

// Prints value of a complex number
void printd(comp val, const char* name) {
	printf("%s: %+.4f%+.4fi\n", name, val.real(), val.imag());
}

// Comparator for sorting an array of flaots using qsort
int compare(const void* a, const void* b) {
   comp f_a = *((comp *) a );
   comp f_b = *((comp *) b );

   if(abs(f_a) == abs(f_b) ) return 0;
   else if ( abs(f_a) < abs(f_b) ) return -1;
   else return 1;
}

// Copies square matrix elements 'from' to 'to'
void copy(comp* from, comp* to, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         to[i*size+j] = from[i*size+j];
      }
   }
}

// Makes matrix A an identity matrix
void eye(comp* A, int size) {
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
void remove_nondiag(comp* A, int size) {
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
double off(comp* A, int size) {
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
double lower(comp * A, int size) {
   double sum = 0;
   for(int i = 0; i < size; i++) {
      for(int j = 0; j <= i - 1; j++) {
         sum += pow(abs(A[i*size+j]),2);
      }
   }

   return sqrt(sum);
}

// Get a vecotr that contains diagonal elements of matrix A
void get_diagonals(comp* d, comp* A, int size) {
   for(int i = 0; i < size; i++) {
      d[i] = A[i*size+i];
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
