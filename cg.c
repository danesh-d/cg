
// This program calculates solution of a matrix equation where 
// the coefficient matrix is symmetric and positive definite, 
// using direct and iterative strategies. The direct solution is 
// aquired from LU solver which is provided by the GNU Scientific 
// Library (GSL). GLS is freely available for Linux distributions. 
// The iterative (Conjugate Gradient) solver is implemented from 
// the scratch. It is possible to feed the program with any equation. 
// At then end, the results from the direct and iterative solvers 
// are shown which can be compared to be very close or same. Under 
// Linux, the code is complied as:
// $ gcc -lgsl -lgslcblas -lm cg.c -o cg

// Written by: Danesh Daroui
// Created: 2012-09-12
// Modified: 2012-11-26


#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

const int n=4;				// matrix dimension in the equation
const float ver=0.83;			// version of the program
const char *date="2012-11-26";		// release date


// show the signature of the program
void sign()
{
	printf("\nWritten by: Danesh Daroui\n");
	printf("Ver. %f\n", ver);
	printf("Release date: %s\n", date);
}


// performs a*x+y operation with two vectors and a scalar
inline void axpy(double *dest, double a, double *x, double *y, int n)
{
	register int i;

	for (i=0; i<n; ++i)
		dest[i]=a*x[i]+y[i];
}


// performs dot product between two vectors
inline double ddot(double *x, double *y, int n)
{
	double final_sum=0;
	register int i;

	for (i=0; i<n; ++i)
		final_sum+=x[i]*y[i];

	return final_sum;
}


// blas level '2' marix-vector operation (i.e. O(n^2) algorithm for a nxn matrix)
inline void blas_dgemv(double *A, double *s, double *z, int nn)
{
	register int i, j;

	for (i=0; i<nn; ++i)
	{
		z[i]=0.0;

		for (j=0; j<nn; ++j)
			z[i]+=A[i*nn+j]*s[j];
	}

	return;
}


// direct solution to "Ax=b" equation, where "A" is a nxn matrix which is 
// performed by calling routines in GSL
void LU_direct(double *A, double *b, double *x, int nn, int mes)
{
	int s=0;	// dummy variable to define sign of the permutation matrix

	// attach matrix "A" and right-hand-side vector "b" to GSL templates
	gsl_matrix_view m=gsl_matrix_view_array(A, nn, nn);
	gsl_vector_view bb=gsl_vector_view_array(b, nn);
	gsl_vector_view xx=gsl_vector_view_array(x, nn);

	// create a permutation matrix to perform reordering, here we do not
	// create an effective permutation matrix, so the matrix is initialized 
	// as an identity matrix which performs no reordering, since the matrix 
	// is just needed for deocmposition and solution of the equation
	gsl_permutation *p=gsl_permutation_calloc(nn);

	// factorize the coefficient matric using LU decomposition
	gsl_linalg_LU_decomp(&m.matrix, p, &s);

	// solve the factorized equation
	gsl_linalg_LU_solve(&m.matrix, p, &bb.vector, &xx.vector);

	// release allocated memory for permutation matrix
	gsl_permutation_free(p);

	// if the "mes" flag is set, then show the notification message
	if (mes)
		printf("-- Solution is done.\n");
}


// conjugate gradient (iterative) solution with preconditioner to "Ax=b" 
// equation, where "A" is a nxn matrix "x" can be either initialized with 
// zero or an initial guess, and "M" is the preconditioner. If a convergence 
// is achieved number of iterations will be returned, other "-1" will be returned
int CG_iterative(double *A, double *b, double *x, double *M, int max_iter, 
			double tol, int nn)
{
	double bnorm2;
	double rnorm2;
	double rz, rzold;
	double alpha, beta;
	double rz_local, rnorm2_local, bnorm2_local;
	double *s;
	double *r;
	double *z;
	double MM[nn*nn];
	register int i;
	register int it;		// 'it' holds the number of iterations that solver has already gone

	// allocate memory for buffer vectors
	s=(double*)malloc(nn*sizeof(double));;
	r=(double*)malloc(nn*sizeof(double));;
	z=(double*)malloc(nn*sizeof(double));;

	// store the preconditioner since it will be replaced by the factors when 
	// the precondtioning equation is solved using direct strategy
	for (i=0; i<nn*nn; ++i)
		MM[i]=M[i];

	bnorm2=ddot(b, b, nn);

	for (i=0; i<nn; ++i)
	{
		x[i]=0.0;		// initial guess is set as zero
		r[i]=b[i];		// r = b
	}

	// apply the preconditioner
	LU_direct(M, r, z, nn, 0);

	// restore the preconditioner
	for (i=0; i<nn*nn; ++i)
		M[i]=MM[i];

	for (i=0; i<nn; ++i)
		s[i]=z[i];

	rz=ddot(r, z, nn);
	rnorm2=ddot(r, r, nn);

	for (it=0; it<max_iter; ++it)
	{
		// perform matrix-vector product in each iteration 
		// and then evaluate the residual
		blas_dgemv(A, s, z, nn);

		alpha=rz/ddot(s, z, n);
		axpy(x, alpha, s, x, n);
		axpy(r, -alpha, z, r, n);

		// apply the preconditioner
		LU_direct(M, r, z, nn, 0);

		for (i=0; i<nn*nn; ++i)
			M[i]=MM[i];

		rzold=rz;

		rz=ddot(r, z, n);
		beta=-rz/rzold;
		axpy(s, -beta, s, z, n);

		// calculate the error and check whether 
		// the solution has been converged or not
		rnorm2=ddot(r, r, n);
		if (rnorm2<=bnorm2*tol*tol)
		{
			printf("-- Solution converged at iteration '%d' with residual norm of %f.\n", it, sqrt(rnorm2));
			break;		// convergence is achieved
		}
	}

	// release the allocated memoy for buffers to avoid memory leak
	free(s);
	free(r);
	free(z);

	// the method did not converge after reaching maximum number of iterations
	if (it>=max_iter)
	{
		printf("-- Solution did not converge at the desired tolerance.\n");
		return -1;
	}

	return it;
}








int main (void)
{
	// coefficient matrix (the matrix is symmetrix and positive definite 
	// since conjugate gradient solver will be used)
	double a_data[]={2.8966, 2.1881, 1.1965, 2.1551,
				2.1881, 1.9966, 0.6827, 1.8861,
				1.1965, 0.6827, 0.7590, 0.5348,
				2.1551, 1.8861, 0.5348, 2.0955};
	// preconditioner (the preconditioner should be, as sparse as possible 
	// to make it easy to be applied. It should also be as close as possible 
	// to the inverse of the coefficient matrix. Here we use no preconditioner 
	// for this equation, because the system is small and the matrix is not 
	// ill-conditioned i.e. condition number after eigenvalues analysis 
	// revealed to be ~20. Therefore, the preconditioner matrix is set to be 
	// as identity matrix)
	double m_data[]={1.0, 0.0, 0.0, 0.0,
				0.0, 1.0, 0.0, 0.0,
				0.0, 0.0, 1.0, 0.0,
				0.0, 0.0, 0.0, 1.0};
	double b_data[]={1.0, 2.0, 3.0, 4.0};		// right-hand-side vector
	double x_direct[]={0.0, 0.0, 0.0, 0.0};		// direct solution
	double x_iterat[]={0.0, 0.0, 0.0, 0.0};		// iterative solution
	double test1[n];				// verification vector
	double test2[n];				// verification vector
	double A[n*n];					// a buffer
	double tol=1e-6;				// stop criteria
	int max_iter=n;					// maximum num of iters
	int iters=0;
	register int i;

	// copy the coefficient matrix to another place, because LU factorization 
	// will be stored over the coefficient matrix
	for (i=0; i<n*n; ++i)
		A[i]=a_data[i];

	sign();

	printf("\n\n\n---------- Solution ----------");
	// solve the equation using direct solver
	LU_direct(A, b_data, x_direct, n, 1);
	printf("Direct (LU) solution results: \n");
	for (i=0; i<n; ++i)
		printf("x[%d]: %f\n", i, x_direct[i]);

	// put some extra empty line between results from different solvers
	printf("\n\n");

	// restore the coefficient matrix for the iterative solver
	for (i=0; i<n*n; ++i)
		A[i]=a_data[i];

	// solve the equation using iterative (conjugate gradient) solver
	iters=CG_iterative(A, b_data, x_iterat, m_data, max_iter, tol, n);
	printf("Iterative (Conjugate Gradient) solution results: \n");
	for (i=0; i<n; ++i)
		printf("x[%d]: %f\n", i, x_iterat[i]);

	// put some extra empty line between results from different solvers
	printf("\n\n\n---------- Verification ----------\n\n");

	printf("Right-hand-side: \n");
	for (i=0; i<n; ++i)
		printf("b[%d]: %f\n", i, b_data[i]);

	// put some extra empty line between results from different solvers
	printf("\n\n");

	blas_dgemv(a_data, x_direct, test1, n);
	printf("Verfying the results from direct solver: \n");
	for (i=0; i<n; ++i)
		printf("b1[%d]: %f\n", i, test1[i]);

	// put some extra empty line between results from different solvers
	printf("\n\n");

	blas_dgemv(a_data, x_iterat, test2, n);
	printf("Verfying the results from iterative solver: \n");
	for (i=0; i<n; ++i)
		printf("b2[%d]: %f\n", i, test2[i]);

	printf("\n");

	return 0;
}

