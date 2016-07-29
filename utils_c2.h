/* Utility functions
 *
 * Author: Basileal Imana
 * Date: 06/13/16
 */

// Libriaries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <complex.h>

#define create_comp(a,b) (a + b*I)
typedef double complex comp;

// Prints a square matrix to stdout
void print(comp* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
      	printf("%+.4f%+.4fi  ", creal(A[i*size+j]), cimag(A[i*size+j]));
		}
      printf("\n");
   }
}

// Prints a square matrix to stdout, column major format
void print_cm(comp* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         printf("%+.4f%+.4fi  ", creal(A[j*size+i]), cimag(A[j*size+i]));
      }
      printf("\n");
   }
}

// Prints a non-square matrix to stdout
void print2(comp* A, int row, int col) {
   for(int i = 0; i < row; i++) {
      for(int j = 0; j < col; j++) {
         printf("%+.4f%+.4fi  ", creal(A[i*col+j]), cimag(A[i*col+j]));
      }
      printf("\n");
   }
}

// Copies square matrix elements 'from' to 'to'
void copy(comp* from, comp* to, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         to[i*size+j] = from[i*size+j];
      }
   }
}

// Makes matrix A an identity matrix - complex
void eye(comp* A, int size) {
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

// Calculates norm of strictly lower triangluar part
// of matrix A
double lower(comp * A, int size) {
   double sum = 0;
   for(int i = 0; i < size; i++) {
      for(int j = 0; j <= i - 1; j++) {
         sum += pow(cabs(A[i*size+j]),2);
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

// Comparator for sorting an array of flaots using qsort
int compare(const void* a, const void* b) {
   comp f_a = *((comp *) a );
   comp f_b = *((comp *) b );

   if(cabs(f_a) == cabs(f_b) ) return 0;
   else if ( cabs(f_a) < cabs(f_b) ) return -1;
   else return 1;
}

// Given pivot(i,j), constructs a submatrix of rows affected J'*A
void create_sub_row(comp* A, int size, int i, int j, comp* A_sub) {
   for(int k = 0; k < size; k++) {
      A_sub[0 * size + k] = A[i * size + k];
      A_sub[1 * size + k] = A[j * size + k];
   }
}

// Given pivot(i,j), constructs a submatrix of row affected by A*J
void create_sub_col(comp* A, int size, int i, int j, comp* A_sub) {
   for(int k = 0; k < size; k++) {
      A_sub[k * 2 + 0] = A[k * size + i];
      A_sub[k * 2 + 1] = A[k * size + j];
   }
}

// Updates the original matrix's rows with changes made to submatrix
void update_sub_row(comp* A, int size, int i, int j, comp* A_sub) {
   for(int k = 0; k < size; k++) {
      A[i * size + k] = A_sub[0 * size + k];
      A[j * size + k] = A_sub[1 * size + k];
   }
}

// Updates the original matrix's cols with changes made to submatrix
void update_sub_col(comp* A, int size, int i, int j, comp* A_sub) {
   for(int k = 0; k < size; k++) {
      A[k * size + i] = A_sub[k * 2 + 0];
      A[k * size + j] = A_sub[k * 2 + 1];
   }
}

// Get ith row of a matrix
void get_ith_row(comp* A, int size, int i, comp* A_i) {
	for(int k = 0; k < size; k++) {
		A_i[k] = A[i*size+k];
	}
}

// Update ith row of a matrix
void update_ith_row(comp* A, int size, int i, comp* A_i) {
	for(int k = 0; k < size; k++) {
		A[i*size+k] = A_i[k];
	}
}

// Get jth column of a matrix
void get_jth_col(comp* A, int size, int j, comp* A_j) {
   for(int k = 0; k < size; k++) {
      A_j[k] = A[k*size+j];
   }
}

// Update jth column of a matrix
void update_jth_col(comp* A, int size, int j, comp* A_j) {
   for(int k = 0; k < size; k++) {
      A[k*size+j] = A_j[k];
   }
}

// Multiplies A(mxk) matrix by B(kxn) matrix
void mul_mat(int m,int n,int k, comp* a, comp* b, comp* c) {
   int i,j,h;
   for(i = 0; i < m; i++) {
      for(j = 0; j < n; j++) {
         c[i * n + j] = 0;
         for(h = 0; h< k; h++) {
            c[i * n + j] += + a[i * k + h] * b[h * n + j];
         }
      }
   }
}

// Multiplies A vector of size n with a scalar c
void vec_mat(int n, comp* V, double c, comp* result) {
	for(int i = 0; i < n; i++) {
		result[i] = V[i] * c;
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
