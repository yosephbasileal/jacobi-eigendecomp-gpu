/* utils.cuh: Utility functions
 *
 * Author: Basileal Imana
 * Date: 06/13/16
 */

// Libriaries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

// Prints a square matrix to stdout
void print(double* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         printf("%.8lf  ",A [i*size+j]);
      }
      printf("\n");
   }
}

// Prints a square matrix to stdout, column major format
void print_cm(double* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         printf("%.8lf  ",A [j*size+i]);
      }
      printf("\n");
   }
}

// Prints a non-square matrix to stdout
void print2(double* A, int row, int col) {
   for(int i = 0; i < row; i++) {
      for(int j = 0; j < col; j++) {
         printf("%.8lf  ", A[i*col+j]);
      }
      printf("\n");
   }
}

// Copies square matrix elements 'from' to 'to'
void copy(double* from, double* to, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         to[i*size+j] = from[i*size+j];
      }
   }
}

// Makes matrix A an identity matrix
void eye(double* A, int size) {
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

// Make all non diagonal elements of square matrix A 0
void remove_nondiag(double* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         if(i != j) {
            A[i*size+j] = 0.0;
         }
      }
   }
}

// Checks if a square matrix is symmetric
bool is_symmetric(double* A, int size) {
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
// all off diagonal elements of symmetric matrix A
double off(double* A, int size) {
   double sum = 0;
   for(int i = 0; i < size - 1; i++) {
      for(int j = i + 1; j < size; j++) {
         // multiply by 2 to account for other half of matrix
         sum += 2 * A[i*size+j] * A[i*size+j];
      }
   }

   return sqrt(sum);
}

// Get a vecotr that contains diagonal elements of matrix A
void get_diagonals(double* d, double* A, int size) {
   for(int i = 0; i < size; i++) {
      d[i] = A[i*size+i];
   }
}

// Comparator for sorting an array of flaots using qsort
int compare(const void* a, const void* b) {
   double f_a = *((double *) a );
   double f_b = *((double *) b );

   if(f_a == f_b ) return 0;
   else if ( f_a < f_b ) return -1;
   else return 1;
}
