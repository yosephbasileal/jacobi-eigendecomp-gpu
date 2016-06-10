
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

bool debug = false; // -d command line option for verbose output

// Prints a matrix to stdout
int print(float* A, int size) {
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			printf("%.4f  ",A [i*size +j]);
		}
		printf("\n");
	}
}

// Copies matrix elements 'from' to 'to'
int copy(float* from, float* to, int size) {
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			to[i*size+j] = from[i*size+j];
		}
	}
}

// Makes matrix A an identity matrix
int eye(float* A, int size) {
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

	while(off(D,size) > epsilon) {
		// execute a cycle of n(n-1)/2 rotations
		for(int i = 0; i < size - 1; i++) {
			for(int j = i + 1; j < size; j++) {
				// calculate values of c and s
				float c, s;
				jacobi_cs(D, size, i, j, &c, &s);

				// setup rotation matrix
				float * R = (float *) malloc(sizeof(float) * size * size);
				eye(R, size);
				R[i*size+i] = c;
				R[j*size+j] = c;
				R[j*size+i] = s;
				R[i*size+j] = -s;

				if(debug) {
					printf("Zeroed out element D(%d,%d)\n",i,j);
				}

				// do rotation
				float* result = (float *) malloc(sizeof(float) * size * size);

				// sgemm calculates C = alpha*A*B + beta*C
				float alpha = 1.0;
				float beta = 0.0;

				// calculate D = R * D * R'
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                    size, size, size, 1.0, R, size, D, size, 0, result, size);
				copy(result,D,size);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, \
                    size, size, size, 1.0, D, size, R, size, 0, result, size);
				copy(result,D,size);

				if(debug) {
					printf("New transformed matrix D:\n");
					print(D,size);
					printf("\n");
				}

				// Cacluate E = E * R
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                    size, size, size, 1.0, E, size, R, size, 0, result, size);
				copy(result,E,size);
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

