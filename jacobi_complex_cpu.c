
/* Jacobi-like method for eigendecomposition of general complex matrices
 *
 * Author: Basileal Imana
 * Date: 07/17/16
 */

// Libriaries
#include <getopt.h>
#include <time.h>
#include "utils_c2.h"

#define eps 0.000000000000001 // 10^-15
#define T 100000000 //10^8
#define M_PI acos(-1.0)

bool debug = false; // -d option for verbose output
bool output = false; // -p option for outputting results
int num_sweeps = 10; // -s option for max number for sweeps

// Calculates parameters for unitary transformation matrix
void unitary_params(comp* A, int size, int p, int q, comp* c, comp* s) {

   comp d_pq, d_max1, d_max2, d_max, m, tan_x, theta, x, e_itheta, e_mitheta;
   double theta_r;

   d_pq = -(A[q*size+q] - A[p*size+p])/2.0;

   d_max1 = d_pq + csqrt(cpow(d_pq,2)+A[p*size+q]*A[q*size+p]);
   d_max2 = d_pq - csqrt(cpow(d_pq,2)+A[p*size+q]*A[q*size+p]);
   d_max = (cabs(d_max1) > cabs(d_max2))? d_max1 : d_max2;

   m = A[q*size+p]/d_max;

   if(cabs(creal(m)) < eps) {
      theta = M_PI/2;
   } else {
      theta = atan(-cimag(m)/creal(m));
   }

   theta_r = creal(theta); //theta is real so take the real part

   e_itheta = create_comp(cos(theta_r),sin(theta_r)); //e^(I * theta)
   e_mitheta = create_comp(cos(theta_r),-sin(theta_r)); //e^(-I * theta)

   tan_x = (e_itheta * A[q*size+p])/d_max;
   x = catan(tan_x);

   *c = ccos(x);
   *s = e_itheta*csin(x);
}

// Calculates parameters for shear transformation matrix
void shear_params(comp* A, int size, int p, int q, comp* c, comp* s) {

   comp g_pq = 0, d_pq, c_pq = 0, e_pq, tanh_y, y, alpha, e_ialpha, e_mialpha, temp;
   double alpha_r;

   for(int j = 0; j < size; j++) {
      if(j != p && j != q) {
         g_pq += cpow(cabs(A[p*size+j]),2) + cpow(cabs(A[q*size+j]),2);
         g_pq += cpow(cabs(A[j*size+p]),2) + cpow(cabs(A[j*size+q]),2);
      }
   }

   d_pq = A[q*size+q] - A[p*size+p];

   for(int j = 0; j < size; j++) {
      c_pq += A[p*size+j]*conj(A[q*size+j]) - conj(A[j*size+p])*A[j*size+q];
   }

   alpha = carg(c_pq) - M_PI/2;
   alpha_r = creal(alpha);

   e_ialpha = create_comp(cos(alpha_r), sin(alpha_r)); //e^(I * alpha)
   e_mialpha = create_comp(cos(alpha_r), -sin(alpha_r)); //e^(-I * alpha)

   e_pq = e_ialpha * A[q*size+p] + e_mialpha * A[p*size+q];

   tanh_y = -cabs(c_pq)/(2*(cpow(cabs(d_pq),2)+cpow(cabs(e_pq),2)) + g_pq);
   y = catanh(tanh_y);

   *c = ccosh(y);
   temp = e_ialpha*csinh(y);
   *s = create_comp(-cimag(temp),creal(temp));
}

// Calculates paramters for diagonal transformation matrix
void diag_params(comp* A, int size, int j, double* t_j) {

   double g_j = 0, h_j = 0;

   for(int l = 0; l < size; l++) {
      if(l != j) {
         g_j += pow(cabs(A[l*size+j]),2);
         h_j += pow(cabs(A[j*size+l]),2);
      }
   }

   g_j = sqrt(g_j);
   h_j = sqrt(h_j);

   *t_j = sqrt(h_j/g_j);
}

// Shear transformation
void shear(comp* A, comp* E, int size, int i, int j) {
	// submatrices (2xn or nx2 size) for storing intermediate results
   comp* A_sub = (comp *) malloc(sizeof(comp) * 2 * size);
   comp* E_sub = (comp *) malloc(sizeof(comp) * 2 * size);
   comp* X_sub = (comp *) malloc(sizeof(comp) * 2 * size);

	comp c, s;
	// calculate values of c and s
   shear_params(A, size, i, j, &c, &s);

   // setup rotation matrix
   comp S[] = {c, -s, create_comp(-creal(s), cimag(s)), c};
   comp S_t[] = {c, s, -create_comp(-creal(s), cimag(s)), c};

   // get submatrix of rows of A that will be affected by S' * A
   create_sub_row(A, size, i, j, A_sub);
   // calculate X_sub = S' * A_sub
   mul_mat(2,size,2,S_t,A_sub,X_sub);
   // update A
   update_sub_row(A,size,i,j,X_sub);

   // get submatrix of cols of A that will be affected by A * S
   create_sub_col(A,size,i,j,A_sub);
   // calculate X_sub = A_sub * S
   mul_mat(size,2,2,A_sub,S,X_sub);
   // update A
   update_sub_col(A,size,i,j,X_sub);

   // get submatrix of cols of E that will be affected by E * S
   create_sub_col(E,size,i,j,E_sub);
   // calculate X_sub = E_sub * S
   mul_mat(size,2,2,E_sub,S,X_sub);
   // update E
   update_sub_col(E,size,i,j,X_sub);

	free(A_sub);
	free(E_sub);
	free(X_sub);
}

// Unitary transformation
void unitary(comp* A, comp* E, int size, int i, int j) {
	// submatrices (2xn or nx2 size) for storing intermediate results
   comp* A_sub = (comp *) malloc(sizeof(comp) * 2 * size);
   comp* E_sub = (comp *) malloc(sizeof(comp) * 2 * size);
   comp* X_sub = (comp *) malloc(sizeof(comp) * 2 * size);

	comp c, s;
   // calculate values of c and s
   unitary_params(A, size, i, j, &c, &s);

   // setup rotation matrix
   comp U[] = {c, -s, -create_comp(-creal(s), cimag(s)), c};
   comp U_t[] = {c, s, create_comp(-creal(s), cimag(s)), c};

   // get submatrix of rows of A that will be affected by U' * A
   create_sub_row(A, size, i, j, A_sub);
   // calculate X_sub = U' * A_sub
   mul_mat(2,size,2,U_t,A_sub,X_sub);
   // update A
   update_sub_row(A,size,i,j,X_sub);

   // get submatrix of cols of A that will be affected by A * U
   create_sub_col(A,size,i,j,A_sub);
   // calculate X_sub = A_sub * U
   mul_mat(size,2,2,A_sub,U,X_sub);
   // update A
   update_sub_col(A,size,i,j,X_sub);

   // get submatrix of cols of E that will be affected by E * S\U
   create_sub_col(E,size,i,j,E_sub);
   // calculate X_sub = E_sub * S\U
   mul_mat(size,2,2,E_sub,U,X_sub);
   // update E
   update_sub_col(E,size,i,j,X_sub);

	free(A_sub);
	free(E_sub);
	free(X_sub);
}

// Diagonal transformation
void diag(comp* A, comp* E, int size, int j) {
   // submatrices (2xn or nx2 size) for storing intermediate results
   comp* A_sub = (comp *) malloc(sizeof(comp) * 2 * size);
   comp* E_sub = (comp *) malloc(sizeof(comp) * 2 * size);
   comp* X_sub = (comp *) malloc(sizeof(comp) * 2 * size);	

	double tj;

	// calculate value tj
   diag_params(A,size,j,&tj);
   // get ith row of A that will be affected by D' * A
   get_ith_row(A,size,j,A_sub);
   // calculate X_sub = 1/tj * A_sub
   vec_mat(size,A_sub,1/tj,X_sub);
   // update A
   update_ith_row(A,size,j,X_sub);

   // get submatrix of cols of A that will be affected by A * D
   get_jth_col(A,size,j,A_sub);
   // calculate X_sub = A_sub * tj
   vec_mat(size,A_sub,tj,X_sub);
   // update A
   update_jth_col(A,size,j,X_sub);

   // get submatrix of cols of E that iwll be affected by E * D
   get_jth_col(E,size,j,E_sub);
   // calculate X_sub = E_sub * tj
   vec_mat(size,E_sub,tj,X_sub);
   // update E
   update_jth_col(E,size,j,X_sub);

	free(A_sub);
	free(E_sub);
	free(X_sub);
}

// Jacobi method
void jacobi(comp* A, comp* E, int size, double epsilon) {
   
	// initialize E
   eye(E, size);

	double cond = (size*size/2) * eps;
   int sweep_count = 0;
   double lowerA;
   
	// do sweeps
   while(((lowerA = lower(A,size)) > cond) && (sweep_count < num_sweeps)) {
      sweep_count++;
      printf("Doing sweep #%d  lower(A) = %.15lf \n", sweep_count, lowerA);
      
		comp c, s;
		double tj;
		// execute a cycle of n(n-1)/2 rotations
      for(int i = 0; i < size - 1; i++) {
         for(int j = i + 1; j < size; j++) {
				shear(A, E, size, i, j);
				unitary(A, E, size, i, j);
			}
			diag(A, E, size, i);
      }
	   diag(A, E, size, size-1);
   }
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
		printf("Error: missing option -N <size of matrix>\n");
		return 0;
	}

   // initialize array
   comp* A = (comp*) malloc(sizeof(comp) * size * size);
   comp* E = (comp*) malloc(sizeof(comp) * size * size);

   // array to store eigenvalues
   comp* ei = (comp *) malloc(sizeof(comp) * size);

	// create a ranom matrix
	create_mat(A, size);

   if(debug) {
		printf("Size: %d\n",size);
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
      	 printf("%+.4f%+.4f\n", creal(ei[i]), cimag(ei[i]));
		}
      printf("\n");
      //printf("Eigenvectors:\n");
      //print(E, size);
      //printf("\n");
   }
	
	printf("\nExecution time: %lf\n", time_spent);

	// clean up
	free(A);
	free(E);
	free(ei);

   return 0;
}
