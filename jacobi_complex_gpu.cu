
/* Jacobi-like method for eigendecomposition of general complex matrices
 *
 * Author: Basileal Imana
 * Date: 07/04/16
 */

// Libriaries
#include <getopt.h>
#include <time.h>
#include "utils_c.cuh"

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

#define eps 0.000000000000001 // 10^-15
#define T 1000000000 // 10^8

bool debug = false; // -d option for verbose output
bool output = false; // -p option for ouputting results
int num_sweeps = 10; // -s option for max number of sweeps

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

__global__ void chess_permute_kernel1(int* order2, const int size) {
	//extern __shared__ int temp[];

	int tid = threadIdx.x;

	int temp = order2[tid+1];
	__syncthreads();
	order2[tid] = temp;
}

__global__ void chess_permute_kernel2(int * order1, const int size) {
	//extern __shared__ int temp[];
	
	int tid = threadIdx.x;

	int temp= order1[tid+1];
	__syncthreads();
	order1[tid+2] = temp;
}


void chess_permute2(int* order1, int* order2, int size) {
    cudaError_t cudaStatus;
	//const int sm_size1 = sizeof(int)*(size-1);
	//const int sm_size2 = sizeof(int)*(size-2);
	int temp = order2[0];

   chess_permute_kernel1<<<1,size-1>>>(order2,size);
	/*for(int i = 0; i <= size - 2; i++) {
      order2[i] = order2[i+1];
   }*/

	cudaStatus = cudaDeviceSynchronize();
   HANDLE_ERROR(cudaStatus);

   order2[size-1] = order1[size-1];

   chess_permute_kernel2<<<1,size-2>>>(order1,size);
   /*for(int i = size - 1; i >= 2; i--) {
      order1[i] = order1[i-1];
   }*/

	   cudaStatus = cudaDeviceSynchronize();
   HANDLE_ERROR(cudaStatus);
	
   order1[1] = temp;
}

// Calculates parameters for unitary transformation matrix
__device__ void unitary_params(comp* A, int size, int p, int q, comp* c, comp* s) {
   
   comp d_pq, d_max1, d_max2, d_max, m, tan_x, theta, x, e_itheta, e_mitheta;
   double theta_r;
	
   d_pq = -(A[q*size+q] - A[p*size+p])/2.0;
   
   d_max1 = d_pq + sqrt(pow(d_pq,2)+A[p*size+q]*A[q*size+p]);
   d_max2 = d_pq - sqrt(pow(d_pq,2)+A[p*size+q]*A[q*size+p]);
   d_max = (abs(d_max1) > abs(d_max2))? d_max1 : d_max2;

   m = A[q*size+p]/d_max;
	
   if(abs(m.real()) < eps) {
      theta = M_PI/2;
   } else {
      theta = atan(-m.imag()/m.real());
   }

   theta_r = theta.real(); //theta is real so take the real part

   e_itheta = create_comp(cos(theta_r),sin(theta_r)); //e^(I * theta)
   e_mitheta = create_comp(cos(theta_r),-sin(theta_r)); //e^(-I * theta)
   
   tan_x = (e_itheta * A[q*size+p])/d_max;
   x = atan(tan_x);

   *c = cos(x);
   *s = e_itheta*sin(x);
}

// Calculates parameters for shear transformation matrix
__device__ void shear_params(comp* A, int size, int p, int q, comp* c, comp* s) {

   comp g_pq = 0, d_pq, c_pq = 0, e_pq, tanh_y, y, alpha, e_ialpha, e_mialpha, temp;
   double alpha_r;

   for(int j = 0; j < size; j++) {
      if(j != p && j != q) {
         g_pq += pow(abs(A[p*size+j]),2) + pow(abs(A[q*size+j]),2);
         g_pq += pow(abs(A[j*size+p]),2) + pow(abs(A[j*size+q]),2);
      }
   }
   
	d_pq = A[q*size+q] - A[p*size+p];

   for(int j = 0; j < size; j++) {
      c_pq += A[p*size+j]*conj(A[q*size+j]) - conj(A[j*size+p])*A[j*size+q];
   }

   alpha = arg(c_pq) - M_PI/2;
   alpha_r = alpha.real();
	
   e_ialpha = create_comp(cos(alpha_r), sin(alpha_r)); //e^(I * alpha)
   e_mialpha = create_comp(cos(alpha_r), -sin(alpha_r)); //e^(-I * alpha)

   e_pq = e_ialpha * A[q*size+p] + e_mialpha * A[p*size+q];

   tanh_y = -abs(c_pq)/(2*(pow(abs(d_pq),2)+pow(abs(e_pq),2)) + g_pq);
	y = atanh(tanh_y);

   *c = cosh(y);
	temp = e_ialpha*sinh(y);
   *s = create_comp(-temp.imag(),temp.real());
}

// Calculates paramters for diagonal transformation matrix
__device__ void diag_params(comp* A, int size, int j, comp* t_j) {
   
   comp g_j, h_j;

   for(int l = 0; l < size; l++) {
      if(l != j) {
         g_j += pow(abs(A[l*size+j]),2);
         h_j += pow(abs(A[j*size+l]),2);
      }
   }

   g_j = sqrt(g_j);
   h_j = sqrt(h_j);

   *t_j = sqrt(h_j/g_j); 
}

// Calculates shear params for all n/2 (i,j) pairs in parallel
__global__ void shear_params_kernel(comp* A, int size, int* arr1, int* arr2, comp* cc, comp* ss) {
	int tid = threadIdx.x;

	int i = arr1[tid];
   int j = arr2[tid];

   if(i > j) {
   	int temp = i;
      i = j;
      j = temp;
   }

   shear_params(A, size, i, j, &cc[tid], &ss[tid]);
}

// Calculates unitary params for all n/2 (i,j) pairs in parallel
__global__ void unitary_params_kernel(comp* A, int size, int* arr1, int* arr2, comp* cc, comp* ss) {
   int tid = threadIdx.x;

   int i = arr1[tid];
   int j = arr2[tid];

   if(i > j) {
      int temp = i;
      i = j;
      j = temp;
   }

   unitary_params(A, size, i, j, &cc[tid], &ss[tid]);
}

// Calculates diag params for all n transformations in parallel
__global__ void diag_params_kernel(comp* A, int size, comp* tj) {
	int tid = threadIdx.x;

	diag_params(A, size, tid, &tj[tid]);
}

// Kerenel 1 for shear transformation
__global__ void jacobi_kernel1_s(comp* A, comp* X, int size, int* arr1, int* arr2, comp* cc, comp* ss) {

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
   comp c = cc[blockIdx.x];
   comp s = ss[blockIdx.x];

   // setup rotation matrices
   comp S_T[] = {c, s, -create_comp(-s.real(),s.imag()), c};

   // get row i and row j elements for current thread
   comp row_i = A[i*size+tid];
   comp row_j = A[j*size+tid];

   // calculate X2 = S' * A, X2 is column major array
   X[tid*size+i] = S_T[0] * row_i + S_T[1] * row_j;
   X[tid*size+j] = S_T[2] * row_i + S_T[3] * row_j;
}

// Kernel 1 for unitary transformation
__global__ void jacobi_kernel1_u(comp* A, comp* X, int size, int* arr1, int* arr2, comp* cc, comp* ss) {

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
   comp c = cc[blockIdx.x];
   comp s = ss[blockIdx.x];

   // setup rotation matrices
   comp U_T[] = {c, s, create_comp(-s.real(),s.imag()), c};

   // get row i and row j elements for current thread
   comp row_i = A[i*size+tid];
   comp row_j = A[j*size+tid];

   // calculate X1 = U' * A, X1 is column major array
   X[tid*size+i] = U_T[0] * row_i + U_T[1] * row_j;
   X[tid*size+j] = U_T[2] * row_i + U_T[3] * row_j;
}

// Kernel 1 for diagonal transformation
__global__ void jacobi_kernel1_d(comp* A, comp* X, int size, comp* tt) {

   int tid = threadIdx.x;

   // all threads in block operate on row j
   int j = blockIdx.x;

   // get precaculated values of t_j for current value j
   comp tj = tt[blockIdx.x];

   // get row j element for current thread
   comp row_j = A[j*size+tid];

   // calculate X = S' * A, X is column major array
   X[tid*size+j] = row_j * (1.0/tj);
}

// Kernel 2 for shear transformation
__global__ void jacobi_kernel2_s(comp* A, comp* E, comp* X, int size, int* arr1, int* arr2, comp* cc, comp* ss) {

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
   comp c = cc[blockIdx.x];
   comp s = ss[blockIdx.x];

   // setup rotation matrices
	comp S[] = {c, -s, create_comp(-s.real(), s.imag()), c};

   // get col i and col j elements of X2 for current thread
   comp x_col_i = X[i*size+tid];
   comp x_col_j = X[j*size+tid];

   // calculate A = X2 * S, X2 is column major array
   A[i*size+tid] = x_col_i * S[0] + x_col_j * S[2];
   A[j*size+tid] = x_col_i * S[1] + x_col_j * S[3];

   // get col i and col j elements of E for current thread
   comp e_col_i = E[i*size+tid];
   comp e_col_j = E[j*size+tid];

   // caclulate E = E * R, E is column major array
   E[i*size+tid] = e_col_i * S[0] + e_col_j * S[2];
   E[j*size+tid] = e_col_i * S[1] + e_col_j * S[3];
}

// Kernel 2 for unitary transformation
__global__ void jacobi_kernel2_u(comp* A, comp* E, comp* X, int size, int* arr1, int* arr2, comp* cc, comp* ss) {

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
   comp c = cc[blockIdx.x];
   comp s = ss[blockIdx.x];

   // setup rotation matrices
   comp U[] = {c, -s, -create_comp(-s.real(),s.imag()), c};

   // get col i and col j elements of X1 for current thread
   comp x_col_i = X[i*size+tid];
   comp x_col_j = X[j*size+tid];

   // calculate A = X1 * U, X1 is column major array
   A[i*size+tid] = x_col_i * U[0] + x_col_j * U[2];
   A[j*size+tid] = x_col_i * U[1] + x_col_j * U[3];

   // get col i and col j elements of E for current thread
   comp e_col_i = E[i*size+tid];
   comp e_col_j = E[j*size+tid];

   // caclulate E = E * R, E is column major array
   E[i*size+tid] = e_col_i * U[0] + e_col_j * U[2];
   E[j*size+tid] = e_col_i * U[1] + e_col_j * U[3];
}

// Kernel 2 for diagonal transformation
__global__ void jacobi_kernel2_d(comp* A, comp* E, comp* X, int size, comp* tt) {

   int tid = threadIdx.x;

   // all threads in block operate on row j
   int j = blockIdx.x;

   // get precaculated values of t_j for current values of j
   comp tj = tt[blockIdx.x];

	// get col j elements of X for current thread
	comp x_col_j = X[j*size+tid];
	
   // calculate X = S' * A, X is column major array
   A[j*size+tid] = x_col_j * tj;

	// get col j element of E for current thread
	comp e_col_j = E[j*size+tid];
	
	// calculate E = E * tj
	E[j*size+tid] = e_col_j * tj;
}

// Jacobi method
void jacobi(comp* A, comp* E, int size, double epsilon) {

   // initialize E
   eye(E, size);

   // device memory pointers for matrices
   comp *A_d, *E_d, *X_d; // E and X are column major arrays

   // chess tournament ordering arr1 stores i, arr2 stroes j
   int *arr1, *arr2;

   // store c and s values for corresponding (i,j) pair
   comp *cc, *ss, *tj;

   cudaError_t cudaStatus;

   // allocate unified memory
   cudaMallocManaged(&arr1, sizeof(int) * size/2);
   cudaMallocManaged(&arr2, sizeof(int) * size/2);
   cudaMallocManaged(&cc, sizeof(comp) * size/2);
   cudaMallocManaged(&ss, sizeof(comp) * size/2);
   cudaMallocManaged(&tj, sizeof(comp) * size);
   cudaMallocManaged(&A_d, sizeof(comp) * size*size);

   // allocate device memory
   cudaMalloc((void **) &E_d, sizeof(comp) * size*size);
   cudaMalloc((void **) &X_d, sizeof(comp) * size*size);

   // copy matrices to device
   copy(A,A_d,size);
   cudaMemcpy(E_d, E, sizeof(comp) * size*size, cudaMemcpyHostToDevice);

	double cond = (size*size/2) * eps;
   int sweep_count = 0;
   double lowerA;
   
	// do sweeps
   while(((lowerA = lower(A_d,size)) > cond) && (sweep_count < num_sweeps)) {

      sweep_count++;
      printf("Doing sweep #%d  lower(A) = %.15lf \n", sweep_count, lowerA);

      // initialize ordering of i,j pairs
      chess_initialize(arr1, arr2, size/2);

      for(int h = 0; h < size-1; h++) {
         // precalcuate parameters for shear transformations
         // so that both kernels use the same rotation matrix
			shear_params_kernel<<<1,size/2>>>(A_d, size, arr1, arr2, cc, ss);

         cudaStatus = cudaDeviceSynchronize();
         HANDLE_ERROR(cudaStatus);

			// launch kernel 1 - shear
         jacobi_kernel1_s<<<size/2,size>>>(A_d, X_d, size, arr1, arr2, cc, ss);

         // synchronize
         cudaStatus = cudaDeviceSynchronize();
         HANDLE_ERROR(cudaStatus);

         // launch kernel 2 - shear
         jacobi_kernel2_s<<<size/2,size>>>(A_d, E_d, X_d, size, arr1, arr2, cc, ss);

			cudaStatus = cudaDeviceSynchronize();
         HANDLE_ERROR(cudaStatus);

			// precalculate parameters for unitary transformations
			unitary_params_kernel<<<1,size/2>>>(A_d, size, arr1, arr2, cc, ss);

         cudaStatus = cudaDeviceSynchronize();
         HANDLE_ERROR(cudaStatus);

			// launch kernel 1 - unitary
         jacobi_kernel1_u<<<size/2,size>>>(A_d, X_d, size, arr1, arr2, cc, ss);

         // synchronize
         cudaStatus = cudaDeviceSynchronize();
         HANDLE_ERROR(cudaStatus);

         // launch kernel 2 - unitary
         jacobi_kernel2_u<<<size/2,size>>>(A_d, E_d, X_d, size, arr1, arr2, cc, ss);

         // synchronize
         cudaStatus = cudaDeviceSynchronize();
         HANDLE_ERROR(cudaStatus);
			
			// precacluate parameters for diagonal transformations
			diag_params_kernel<<<1,size>>>(A_d, size, tj);
			
			cudaStatus = cudaDeviceSynchronize();
         HANDLE_ERROR(cudaStatus);

			// launch kernel 1 - diagonal
			jacobi_kernel1_d<<<size,size>>>(A_d, X_d, size, tj);

			// synchronize
			cudaStatus = cudaDeviceSynchronize();
			HANDLE_ERROR(cudaStatus);

			// launch kernel 2 - diagonal
			jacobi_kernel2_d<<<size,size>>>(A_d, E_d, X_d, size, tj);

			// synchronize
			cudaStatus = cudaDeviceSynchronize();	
			HANDLE_ERROR(cudaStatus);
		
         // do next permutation of i, j pairs
         chess_permute(arr1, arr2, size/2);
      }

      if(debug) {
         printf("One sweep done. New matrix D: \n");
         print(A_d, size);
         printf("\n");
      }
   }

   // copy to host
   copy(A_d,A,size);
   cudaMemcpy(E, E_d, sizeof(comp) * size*size, cudaMemcpyDeviceToHost);

   // free memory
   cudaFree(arr1);
   cudaFree(arr2);
   cudaFree(cc);
   cudaFree(ss);
   cudaFree(tj);
   cudaFree(A_d);
   cudaFree(E_d);
   cudaFree(X_d);
}

// Main
int main(int argc, char** argv) {

   // process command line arguments
   int r;
	int size = 0;
   while ((r = getopt(argc, argv, "dpN:s:")) != -1) {
      switch(r)
      {
         case 'd':
            debug = true;
            break;
         case 'p':
            output = true;
            break;
         case 'N':
            size = atoi(optarg);
            break;
			case 's':
				num_sweeps = atoi(optarg);
				break;
         default:
            exit(1);
      }
   }

   if(size == 0) {
      printf("Error: missing option -N <size of matrix>)\n");
      return 0;
   }

   // initialize arrays
   comp* A = (comp*) malloc(sizeof(comp) * size * size);
   comp* E = (comp*) malloc(sizeof(comp) * size * size);

   // array to store eigenvalues
   comp* ei = (comp*) malloc(sizeof(comp) * size);

   // create a random matrix
   create_mat(A, size);

   if(debug) {
      printf("Input matrix A: \n");
      print(A, size);
      printf("\n");
   }

   clock_t begin, end;
   double time_spent;

   begin = clock();

   // call facobi method
   jacobi(A, E, size, eps);

   end = clock();
   time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

   get_diagonals(ei, A, size);
   qsort(ei, size, sizeof(comp), compare);

   // output results
   if(output) {
      printf("\n");
      printf("Eigenvalues:\n");
      for(int i = 0; i < size; i++) {
         printf("%+.4f%+.4f\n", ei[i].real(), ei[i].imag());
      }
      printf("\n");
      //printf("Eigenvectors:\n");
      //print_cm(E, size);
      //printf("\n");
   }

   printf("Execution time: %lf\n", time_spent);

   // clean up
   free(A);
   free(E);
   free(ei);

   return 0;
}

