#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <x86intrin.h>

#include "timer.c"

#define N_ 4096
#define K_ 4096
#define M_ 4096
//#define DEBUG_MODE

typedef double dtype;

//works best on debugging square matrices  
void printDebug(dtype *A, dtype *B, dtype *C, int col, int row, int N){
	if(N > 10) {
		printf("No one wants to print that many numbers\n");
		return;
	}
	printf("\nA row: ");
	for(int i=0;i<N;i++){
		printf("%f ||==|| ",A[row*N+i]);
	}
	printf("\nB col: ");
	for(int j=0;j<N;j++){
		printf("%f ||==|| ",B[j*N+col]);
	}
	printf("\nC ans: %f\n",C[col,row]);
}

void verify(dtype *C, dtype *C_ans, int N, int M)
{
  int i, cnt;
  cnt = 0;
  for(i = 0; i < N * M; i++) {
    if(abs (C[i] - C_ans[i]) > 1e-6) cnt++;
  }
  if(cnt != 0) printf("ERROR of margin %d\n",cnt); else printf("SUCCESS\n");
}

void mm_serial (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
  int i, j, k;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      for(int k = 0; k < K; k++) {
        C[i * M + j] += A[i * K + k] * B[k * M + j];
      }
    }
  }
}

void mm_cb (dtype *C, dtype *A, dtype *B, int N, int K, int M, int S)
{
  /* =======================================================+ */
  /* Implement your own cache-blocked matrix-matrix multiply  */
  /* =======================================================+ */
  //S is the bsize, length of one side of a subblock 
  //First version of the code ia assuming all arrays are of equal size and are squared 
  int en = N; //en is how many columns can fit evenly inside of the subblocks 
  
  //printf("N size: %d === B size: %d\n",N,S);
  for(int kk=0;kk<en;kk += S){
      for(int jj=0;jj<en;jj += S){
		  for(int i=0;i<N;i++){
			  for(int j = jj;j<jj+S;j++){
			      double sum = C[i*N+j];
				  for(int k = kk;k<kk+S;k++) 
					  sum += A[i*N+k]*B[k*N+j];
				  C[i*N+j] = sum;
			  }
		  }
	  }
  }
}

void mm_sv (dtype *C, dtype *A, dtype *B, int N, int K, int M,int S)
{
  /* =======================================================+ */
  /* Implement your own SIMD-vectorized matrix-matrix multiply  */
  /* =======================================================+ */
  int en = N; //en is how many columns can fit evenly inside of the subblocks 
  __m128d sum,product,product2,Atemp,Btemp,Btemp2,sumFinal;
  double tempCol[2],tempCol2[2];
  
  //printf("N size: %d === B size: %d\n",N,S);
  for(int kk=0;kk<en;kk += S){
      for(int jj=0;jj<en;jj += S){
		  for(int i=0;i<N;i++){
			  for(int j = jj;j<jj+S;j+=2){
				  sum = _mm_load_pd(&C[i*N+j]);
				  for(int k = kk;k<kk+S;k+=2){
					  Atemp = _mm_load_pd(&A[i*N+k]);
					  
					  //tempCol[0] = B[k*N+j];
					  //tempCol[1] = B[(k+1)*N+j];
					  //Btemp = _mm_load_pd(&tempCol[0]);
					  Btemp = _mm_set_pd(B[(k+1)*N+j],B[k*N+j]);
					  
					  //tempCol2[0] = B[k*N+j+1];
					  //tempCol2[1] = B[(k+1)*N+j+1];
					  //Btemp2 = _mm_load_pd(&tempCol2[0]);
					  Btemp2 = _mm_set_pd(B[(k+1)*N+j+1],B[k*N+j+1]);
					  
					  product  = _mm_mul_pd(Atemp,Btemp);
					  product2 = _mm_mul_pd(Atemp,Btemp2); 
					  sumFinal = _mm_hadd_pd(product,product2);
					  sum = _mm_add_pd(sum,sumFinal);
				  }
				  _mm_store_pd(&C[i*N+j],sum);
			  }
		  }
	  }
  }
}

int main(int argc, char** argv)
{
  int i, j, k;
  int N, K, M;

  if(argc == 4) {
    N = atoi (argv[1]);		
    K = atoi (argv[2]);		
    M = atoi (argv[3]);		
    printf("N: %d K: %d M: %d\n", N, K, M);
  } else {
    N = N_;
    K = K_;
    M = M_;
    printf("N: %d K: %d M: %d\n", N, K, M);	
  }

  dtype *A = (dtype*) malloc (N * K * sizeof (dtype));
  dtype *B = (dtype*) malloc (K * M * sizeof (dtype));
  dtype *C = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_cb = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_sv = (dtype*) malloc (N * M * sizeof (dtype));
  assert (A && B && C);

  /* initialize A, B, C */
  srand48 (time (NULL));
  for(i = 0; i < N; i++) {
    for(j = 0; j < K; j++) {
      A[i * K + j] = drand48 ();
    }
  }
  for(i = 0; i < K; i++) {
    for(j = 0; j < M; j++) {
      B[i * M + j] = drand48 ();
    }
  }
  bzero(C, N * M * sizeof (dtype));
  bzero(C_cb, N * M * sizeof (dtype));
  bzero(C_sv, N * M * sizeof (dtype));

  stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create ();
  assert (timer);
  long double t;

  printf("Naive matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_serial (C, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for naive implementation: %Lg seconds\n\n", t);


  printf("Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_cb (C_cb, A, B, N, K, M,N/4);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for cache-blocked implementation: %Lg seconds\n", t);
  #ifdef DEBUG_MODE
	printDebug(A,B,C_cb,1,1,N);
  #endif

  /* verify answer */
  verify (C_cb, C, N, M);

  printf("SIMD-vectorized Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  #ifdef DEBUG_MODE
	//assuming arguements of 8 8 8
	A[0] = 5.0;A[1] = 5.0;A[3] = 3.0;A[4] = 3.0;
	
	B[0] = 4.0;B[1] = 6.0;
	B[8] = 4.0;B[9] = 6.0;
  #endif
  mm_sv (C_sv, A, B, N, K, M,N/4);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for SIMD-vectorized cache-blocked implementation: %Lg seconds\n", t);
  #ifdef DEBUG_MODE
	printDebug(A,B,C_sv,1,1,N);
  #endif 

  /* verify answer */
  verify (C_sv, C, N, M);

  return 0;
}
