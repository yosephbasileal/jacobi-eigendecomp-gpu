#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <lapacke.h>
#include <cblas.h>


// Needed compile flags: -llapacke -lblas -lm

// Prints a matrix to stdout
int print(float* A, int size) {
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			printf("%f  ",A [i * size + j]);
		}
		printf("\n");
	}
}

// Copies matrix elements 'from' to 'to'
int copy(float* from, float* to, int size) {
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			to[i * size + j] = from[i * size + j];
		}
	}
}

// Makes matrix A an identity matrix
int eye(float* A, int size) {
	for(int i = 0; i < size; i++) {
		for(int j = i; j < size; j++) {
			if(i == j) {
				A[i * size + j] = 1.0;
			} else {
				A[i * size + j] = 0.0;
				A[j * size + i] = 0.0;
			}
		}
	}
}

// Checks if matrix is symmetric
bool is_symmetric(float* A, int size) {
	for(int i = 0; i < size - 1; i++) {
		for(int j = i + 1; j < size; j++) {
			if(A[i * size + j] != A[j * size + i]) {
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
//	all off diagonal elements of A
float off(float* A, int size) {
	float sum = 0;
	for(int i = 0; i < size - 1; i++) {
		for(int j = i + 1; j < size; j++) {
			// multiply by 2 to account for other half of matrix
			sum += 2 * square(A[i * size + j]);
		}
	}

	return sqrt(sum);
}

// Gives the number stored at row'th row and col'th col of A
float index(float* A, int size, int row, int col) {
	return A[row * size + col];
}

// Modiy the number stored at row'th row and col'th col of A
void index_m(float* A, int size, int row, int col, float new_val) {
	A[row * size + col] = new_val;
}
// Cacluates values of c and s for a given pivot of rotation (i,j)
void jacobi_cs(float* A, int size, int i, int j, float* c, float* s) {
	// calculate T
	float T = (index(A,size,j,j) - index(A,size,i,i)) / (2 * index(A,size,i,j));
 
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
		for(int i = 0; i < size/2 - 1; i++) {
			for(int j = i + 1; j < size/2; j++) {
				
				// calculate values of c and s
				float c, s;
				jacobi_cs(A, size, i, j, &c, &s);

				// setup rotation matrix
				float * R = (float *) malloc(sizeof(float) * size * size);
				eye(R, size);
				index_m(R,size,i,i,c);
				index_m(R,size,j,j,c);
				index_m(R,size,j,i,s);
				index_m(R,size,i,j,-s);
				printf("i: %d  j: %d \n",i,j);
				print(R,size);	
				// do rotation
				float* result = (float *) malloc(sizeof(float) * size * size);
				cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,size,size,size,1.0,R,size,D,size,1.0,result,size);
				copy(result,D,size);
				
				print(R,size);
				print(D,size);
				printf("\n");


				cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,size,size,size, 1.0,D,size,R,size,1.0,result,size);
				copy(result,D,size);

				print(D,size);
				printf("\n");
			}
		}
		break;
	}

}
int main() {
	// initialize array
	int size = 6;
	float* A = (float*) malloc(sizeof(float) * size * size);
	float* D = (float*) malloc(sizeof(float) * size * size);
	float* E = (float*) malloc(sizeof(float) * size * size);

	// read matrix from file
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			scanf("%f", &A[i * size + j]);
		}
	}
	print(A, size);
	printf("\n");

	float epsilon = 0.01;	
	jacobi(A, D, E, size, epsilon);

	return 0;
}



