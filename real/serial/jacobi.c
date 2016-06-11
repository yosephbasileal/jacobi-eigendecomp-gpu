
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
#include <cblas.h>
#include <lapacke.h>

bool debug = false; // -d command line option for verbose output

// Prints a matrix to stdout
void print(float* A, int size) {
   for(int i = 0; i < size; i++) {
      for(int j = 0; j < size; j++) {
         printf("%.4f  ", A[i*size+j]);
      }
      printf("\n");
   }
}

void print2(float* A, int row, int col) {
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

// Squares a number
float square(float num) {
   return num * num;
}

// Calculates the square root of sum of squares of
// all off diagonal elements of A
float off(float* A, int size) {
   float sum = 0;
   for(int i = 0; i < size - 1; i++) {
      for(int j = i + 1; j < size; j++) {
         // multiply by 2 to account for other half of matrix
         sum += 2 * square(A[i*size+j]);
      }
   }

   return sqrt(sum);
}

// Given pivot(i,j), constructs a submatrix of rows affected J'*A
void create_sub_row(float* A, int size, int i, int j, float* A_sub) {
   for(int k = 0; k < size; k++) {
      A_sub[0 * size + k] = A[i * size + k];
      A_sub[1 * size + k] = A[j * size + k];
   }
}

// Given pivot(i,j), constructs a submatrix of row affected by A*J
void create_sub_col(float* A, int size, int i, int j, float* A_sub) {
   for(int k = 0; k < size; k++) {
      A_sub[k * size + 0] = A[k * size + i];
      A_sub[k * size + 1] = A[k * size + j];
   }
}

// Updates the original matrix's rows with changes made to submatrix
void update_sub_row(float* A, int size, int i, int j, float* A_sub) {
   for(int k = 0; k < size; k++) {
      A[i * size + k] = A_sub[0 * size + k];
      A[j * size + k] = A_sub[1 * size + k];
   }
}

// Updates the original matrix's cols with changes made to submatrix
void update_sub_col(float* A, int size, int i, int j, float* A_sub) {
   for(int k = 0; k < size; k++) {
      A[k * size + i] = A_sub[k * size + 0];
      A[k * size + j] = A_sub[k * size + 1];
   }
}


// Cacluates values of c and s for a given pivot of rotation (i,j)
void jacobi_cs(float* A, int size, int i, int j, float* c, float* s) {
   // calculate T
   float T = (A[j*size+j] - A[i*size+i]) / (2 * A[i*size+j]);

   // equation: t^2 + 2Tt - 1 = 0
   // chose the root that is smaller in absolute value
   float t;
   if(T >= 0) {
      t = -T + sqrt(1.0 + square(T));
   } else {
      t = -T - sqrt(1.0 + square(T));
   }

   // calculate c and s
   *c = 1.0 / (sqrt(1.0 + square(t)));
   *s = *c * t;
}


// Jacobi method
void jacobi(float* A, float* D, float* E, int size, float epsilon) {
   // initialize D and E
   copy(A, D, size);
   eye(E, size);

	// Submatrices (2xn or nx2 size) for storing intermediate results
	float* D_sub = (float *) malloc(sizeof(float) * 2 * size);
	float* E_sub = (float *) malloc(sizeof(float) * 2 * size);
	float* X_sub = (float *) malloc(sizeof(float) * 2 * size);


   while(off(D,size) > epsilon) {
      // execute a cycle of n(n-1)/2 rotations
      for(int i = 0; i < size - 1; i++) {
         for(int j = i + 1; j < size; j++) {
            // calculate values of c and s
            float c, s;
            jacobi_cs(D, size, i, j, &c, &s);

            // setup rotation matrix
            float R[] = {c, s, -s, c};

            if(debug) {
               printf("Zeroed out element D(%d,%d)\n",i,j);
            }

            // get submatrix of rows of D that will be affected by R' * D
            create_sub_row(D, size, i, j, D_sub);

            // sgemm calculates C = alpha*A*B + beta*C
            float alpha = 1.0;
            float beta = 0.0;

            // calculate X_sub = R' * D_sub
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, \
                    2, size, 2, alpha, R, 2, D_sub, size, beta, X_sub, size);

				// update D
            update_sub_row(D,size,i,j,X_sub);

				// get submatrix of cols of D that will be affected by D * R
            create_sub_col(D,size,i,j,D_sub);

				// calculate X_sub = D_sub * R
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                    size, 2, 2, alpha, D_sub, size, R, 2, beta, X_sub, size);

				// update D
            update_sub_col(D,size,i,j,X_sub);

            if(debug) {
               printf("New transformed matrix D:\n");
               print(D,size);
               printf("\n");
            }

            // get submatrix of cols of E that iwll be affected by E * R
				create_sub_col(E,size,i,j,E_sub);

				// calculate X_sub = E_sub * R
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                    size, 2, 2, alpha, E_sub, size, R, 2, beta, X_sub, size);
            
				// update E
				update_sub_col(E,size,i,j,X_sub);
         }
      }
   }
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
   float epsilon = 0.001;

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

   return 0;
}

