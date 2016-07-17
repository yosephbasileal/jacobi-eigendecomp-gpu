
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

bool debug = false; // -d command line option for verbose output
bool output = false; // -p command line option for outputting results

double epsilon = 0.01; // -e command line option for desired accuracy
int num_sweeps = 6; // -s command line option for number of sweeps

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

// Jacobi method
void jacobi(double* A, double* D, double* E, int size, double epsilon, int num_sweeps) {
   printf("Initializing jacobi matrices...\n");
   // initialize D and E
   copy(A, D, size);
   eye(E, size);

   // submatrices (2xn or nx2 size) for storing intermediate results
   double* D_sub = (double *) malloc(sizeof(double) * 2 * size);
   double* E_sub = (double *) malloc(sizeof(double) * 2 * size);
   double* X_sub = (double *) malloc(sizeof(double) * 2 * size);

   int sweep_count = 0;
   double offA;
   // do sweeps
   while((offA = off(D,size)) > epsilon && (sweep_count < num_sweeps)) {
      sweep_count++;
      printf("Doing sweep #%d  off(D) = %.8lf \n", sweep_count, offA);
      // execute a cycle of n(n-1)/2 rotations
      for(int i = 0; i < size - 1; i++) {
         for(int j = i + 1; j < size; j++) {
            // calculate values of c and s
            double c, s;
            jacobi_cs(D, size, i, j, &c, &s);

            // setup rotation matrix
            double R[] = {c, s, -s, c};
            double R_t[] = {c, -s, s, c};

            if(debug) {
               printf("Zeroed out element D(%d,%d)\n",i,j);
            }

            // get submatrix of rows of D that will be affected by R' * D
            create_sub_row(D, size, i, j, D_sub);

            // sgemm calculates C = alpha*A*B + beta*C
            double alpha = 1.0;
            double beta = 0.0;

            // calculate X_sub = R' * D_sub
            //cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, \
                    2, size, 2, alpha, R, 2, D_sub, size, beta, X_sub, size);
            mul_mat(2,size,2,R_t,D_sub,X_sub);

            // update D
            update_sub_row(D,size,i,j,X_sub);

            // get submatrix of cols of D that will be affected by D * R
            create_sub_col(D,size,i,j,D_sub);

            // calculate X_sub = D_sub * R
            //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                    size, 2, 2, alpha, D_sub, size, R, 2, beta, X_sub, size);
            mul_mat(size,2,2,D_sub,R,X_sub);

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
            //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                    size, 2, 2, alpha, E_sub, size, R, 2, beta, X_sub, size);
            mul_mat(size,2,2,E_sub,R,X_sub);

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

   // initialize array
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
         printf("%.8lf\n", ei[i]);
      }
      printf("\n");
      //printf("Eigenvectors:\n");
      //print(E, size);
      //printf("\n");
   }
	
	printf("Execution time of Jacobi: %lf\n", time_spent);

	// clean up
	free(A);
	free(D);
	free(E);

   return 0;
}
