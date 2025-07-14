// https://github.com/maykhiago/QWBench
// set PATH=C:\Program Files\gcc64\bin;%PATH%#include <windows.h>
// Uso: ./coined.exe entradasCoined_05_25/30C.txt 1 0 2
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <mkl.h>

//#define CLOCK_MONOTONIC 0

void disable_stdout_buffering() {
    setbuf(stdout, NULL);
}

/* BEGIN: neblina.h */
typedef union __data_vector_u {
    void               * v;
    int                * i;
    double              * f;
    void              ** s;
} data_vector_u;

typedef enum  {
        T_STRING,
        T_INT,
        T_FLOAT,
        T_COMPLEX,
        T_ADDR,
        T_NDEF,
        T_LIST,
        T_STRTOREL,
        T_CFUNC,
        T_VECTOR,
        T_MATRIX,
        T_SMATRIX,
        T_RMATRIX,
        T_FILE,
        T_ANY
} data_type;
/* END: neblina.h */

/* BEGIN: neblina_matrix.h */
typedef struct __matrix_t {
    data_vector_u      value;
    int                ncol;
    int                nrow;
    data_type          type;
    unsigned char    location;
    void*             extra;
    int                externalData;
} matrix_t;
/* END: neblina_matrix.h */

/* BEGIN: neblina_smatrix.h */
typedef struct __smatrix_t {

    int nrow;
    int ncol;
    int nnz;

    int* row_ptr;
    int* col_idx;
    double* values;

    int isPacked;
           
    data_type        type;
    unsigned char    location;
    void*           extra;
    void*           idxColMem;
   
} smatrix_t;
/* END: neblina_smatrix.h */


/* BEGIN: neblina_vector.h */
typedef struct __vector_t {
    data_vector_u      value;
    int                len;
    data_type          type;
    unsigned char      location;
    void*              extra;
    int                externalData;
} vector_t;
/* END: neblina_vector.h */

#include "lerDados.h"
void print_vectorR(const vector_t *v, int inicio_, int fim_);

/* BEGIN: libneblina-cpu-bridge-vector.c */
vector_t * vector_new( int len, data_type type, int initialize, void * data ) {
//    printf("em vector_new, len=%d \n", len);
    vector_t * ret = (vector_t *) malloc( sizeof( vector_t ) );
    if (initialize && data == NULL) {
        if( type == T_INT ) {
            ret->value.i = (int *) malloc( len * sizeof( int ) );
        } else if( type == T_FLOAT ){
            ret->value.f = (double *) malloc( len * sizeof( double ) );
}
        else if( type == T_COMPLEX )
            ret->value.f = (double *) malloc( 2 * len * sizeof( double ) );
        ret->externalData = 0;
    } else if (data != NULL) {
        ret->value.f = (double *)data;
        ret->externalData = 1;
    } else {
        ret->value.f = NULL;
        ret->externalData = 0;
    }
    ret->type      = type;
    ret->len       = len;
    //ret->location  = LOCHOS;
    ret->extra       = NULL;
    return ret;
}
/* END: libneblina-cpu-bridge-vector.c */

/* BEGIN: libneblina-cpu-bridge-smatrix.c */

smatrix_t * smatrix_new(int nrow, int ncol, data_type type) {
    smatrix_t *smatrix = (smatrix_t *) malloc(sizeof(smatrix_t));
    smatrix->ncol = ncol;
    smatrix->nrow = nrow;
    smatrix->type = type;
    smatrix->nnz  = 0;
   
    smatrix->row_ptr  = NULL;
    smatrix->col_idx  = NULL;
    smatrix->values   = NULL;
    smatrix->isPacked = 0;
   
    return smatrix;
}

void smatrix_delete(smatrix_t *smatrix) {
    if (!smatrix) { return; }

    // Free CSR-related arrays if allocated
    free(smatrix->row_ptr);
    smatrix->row_ptr = NULL;

    return; // BD abril.25

    free(smatrix->col_idx);
    smatrix->col_idx = NULL;

    free(smatrix->values);
    smatrix->values = NULL;

    // Free additional pointers if allocated
    free(smatrix->extra); smatrix->extra = NULL;

    free(smatrix->idxColMem); smatrix->idxColMem = NULL;

    // Finally, free the struct itself
    free(smatrix);
}

void compute_U_row_ptr(const smatrix_t* S, const smatrix_t* C, smatrix_t* U) {
    U->row_ptr[0] = 0; // First row starts at index 0
    for (int i = 0; i < S->nrow; i++) {
        int permuted_row = S->col_idx[i];  // Get the row index in C
        int nnz_in_C_row = C->row_ptr[permuted_row + 1] - C->row_ptr[permuted_row];
        U->row_ptr[i + 1] = U->row_ptr[i] + nnz_in_C_row;
    }
}

void permute_and_multiply(const smatrix_t* S, const smatrix_t* C, smatrix_t* U) {
    //printf(" em permute_and_multiply, \n");
    //printf("S->type = %d, C->type = %d, U->type = %d\n", S->type, C->type, U->type);
    if (S->type == T_COMPLEX && C->type == T_COMPLEX && U->type == T_COMPLEX) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < S->nrow; ++row) {
            int permuted_row = S->col_idx[row]; // Since S is a permutation matrix
            int startC = C->row_ptr[permuted_row];
            int endC   = C->row_ptr[permuted_row + 1];
            int startU = U->row_ptr[row];
            int j = C->col_idx[startC];

            for (int i = 0; i < (endC - startC); i++) {
                U->col_idx[startU + i] = j++;
                U->values[2 * (startU + i)] = C->values[2 * (startC + i)];
                U->values[2 * (startU + i) + 1] = C->values[2 * (startC + i) + 1];
            }
        }
    } else if (S->type == T_FLOAT && C->type == T_FLOAT && U->type == T_FLOAT) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < S->nrow; ++row) {
            int permuted_row = S->col_idx[row]; // Since A is a permutation matrix
            int startC = C->row_ptr[permuted_row];
            int endC   = C->row_ptr[permuted_row + 1];
            int startU = U->row_ptr[row];
            //    print_smatrix(C); exit(0);
            //Only for block diagonal matrices
            int j = C->col_idx[startC];

            for (int i = 0; i < (endC - startC); i++) {
            //   printf(">> %d - %d\n", C->col_idx[startC + i], j);
                U->col_idx[startU + i] = j++; //C->col_idx[startC + i];
                U->values[startU + i] = C->values[startC + i];
            }
        }
    } else {
        fprintf(stderr, "Incompatible types in permute_and_multiply\n");
        exit(EXIT_FAILURE);
    }
}
/* END: libneblina-cpu-bridge-smatrix.c */

/* BEGIN: libneblina-cpu-bridge-vector.c */
void vector_delete( vector_t * v ) {
    if (v != NULL) {
        if (v->value.f != NULL && v->externalData == 0) {
            free(v->value.f);
        }
        if (v->extra != NULL) {  // No need to check externalData for extra
            free(v->extra);
        }
        free(v);
    }
}
/* END: libneblina-cpu-bridge-vector.c */


/* MAIN CODE FOR TESTING */

void allocate_result(smatrix_t *p, smatrix_t *d, smatrix_t *r){
    r->row_ptr = (int*)    malloc((p->nrow + 1) * sizeof(int));
    r->col_idx = (int*)    malloc((d->nnz)      * sizeof(int));
    r->type    = d->type;

    if(r->type == T_COMPLEX)
        r->values  = (double*) malloc((2*d->nnz)   * sizeof(double));
    else
        r->values  = (double*) malloc((d->nnz)      * sizeof(double));
   
    r->nnz = d->nnz; // r has the same stricture as d
   
    if (!r->row_ptr || !r->col_idx || !r->values) {
        printf("Erro: Failing to allocate memory for r.\n");
        exit(0);
    }
}

void smatrix_vector_mult(const smatrix_t* S, const vector_t* v, vector_t* result) {
//printf("em smatrix_vector_mult,\n ");
    //printf("S->type = %d, v->type = %d, result->type = %d\n", S->type, v->type, result->type);
    if (S->type == T_COMPLEX && v->type == T_COMPLEX && result->type == T_COMPLEX) {
//schedule(runtime)
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < S->nrow; row++) {
            int start = S->row_ptr[row], end = S->row_ptr[row + 1];
                int   k = S->col_idx[start];
            double aSum = 0.0, bSum = 0.0;
            for (int j = start; j < end; j++) {
                double aS = S-> values[2*j];
                double bS = S-> values[2*j+1];
                double aV = v->value.f[2*k];    // Usar col, não k
                double bV = v->value.f[2*k +1];  // Usar col, não k

 //               double aV = v->value.f[ 2 * S->col_idx[j] ];    // Usar col, não k
  //              double bV = v->value.f[ 2 * S->col_idx[j] + 1 ];  // Usar col, não k
		k++;
                double aP = aS * aV - bS * bV;
                double bP = aS * bV + bS * aV;  
                aSum += aP; bSum += bP;
            }
            result->value.f[2*row]   = aSum;
            result->value.f[2*row+1] = bSum;
        }

    } else if (S->type == T_FLOAT && v->type == T_FLOAT && result->type == T_FLOAT) {
// schedule(runtime)
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < S->nrow; row++) {
        int start = S->row_ptr[row], end = S->row_ptr[row + 1];
        int     k = S->col_idx[start];
        double sum =  (0.0*row)/S->nrow;
        for (int j = start; j < end; j++) {
            sum += S->values[j] * v->value.f[k++];
        }
        result->value.f[row] = sum;;
    }
    } else {
        fprintf(stderr, "Incompatible types in smatrix_vector_mult\n");
        exit(EXIT_FAILURE);
    }
}

void permute_vector(const smatrix_t* S, const vector_t* v, vector_t* result) {
//printf(" em permute_vector, \n");
    if (S->type == T_COMPLEX && v->type == T_COMPLEX && result->type == T_COMPLEX) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < S->nrow; ++row) {
            int col = S->col_idx[S->row_ptr[row]]; // Only one entry per row
            // Copy real and imaginary parts of the complex number
            result->value.f[2 * row]     = v->value.f[2 * col];     // real part
            result->value.f[2 * row + 1] = v->value.f[2 * col + 1]; // imaginary part
        }
    } else if (S->type == T_FLOAT && v->type == T_FLOAT && result->type == T_FLOAT) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < S->nrow; ++row) {
            int col = S->col_idx[S->row_ptr[row]]; // Only one entry per row
            result->value.f[row] = v->value.f[col]; // Just permute the value
        }
    } else {
        fprintf(stderr, "Incompatible types in permute_vector\n");
        exit(EXIT_FAILURE);
    }
}

int ARGC; char **ARGV;
char *arqEntrada;//="entradasCoined_04_25/177R.txt";
int REPETICOES=1;//4000;
int REPETICOESG=1;//4000;
int numIterations=1;
bool computeU;

void printC(const double a_, const double b_){ printf("(%7.2e ", a_); printf("%+7.2ei)",  b_ ); }

void printRp(const double *a_){ printf("%9.2e ", *a_); }
void printCp(const double *c_){ printf("(%9.2e ", *c_);c_++; printf("%+9.2ei)",  *c_ ); }

//void (*print_tVector) (const vector_t *,  int , int);
void (*print_tVector) (const vector_t *,  int , int) = print_vectorR;
//void (*printElemntP) (const double *);
void (*printElemntP) (const double *) = printRp;

void print_vectorR(const vector_t *v, int inicio_, int fim_);
void print_vectorC(const vector_t *v, int inicio_, int fim_);
void mostrarConteudoV(vector_t *v, char* label_) ;
void print_dense(const smatrix_t* mat) ;
void print_smatrix(const smatrix_t* matrix);


char * labelComputeU="algebra via   MKL: U=mkl_sparse_spmm(S_, C_); y=mkl_sparse_spmv(U, x) \n";
void simulateMKL(smatrix_t *S, smatrix_t *C, smatrix_t *U_, vector_t *v) {

    #if (!defined(__INTEL_LLVM_COMPILER) &&  !defined(__INTEL_COMPILER))
        fprintf(stderr, "MKL não incluida. \n");
        fprintf(stderr, "Compilado com GCC Compiler. \n");
  //  #else
   //     printf("Compilado com Intel C Compiler (ICC)\n");


    struct timespec startP, endP, startI, endI;
    double tempo_decorridoP = 0.0, tempo_decorridoI = 0.0;


    int aux = (C->type == T_COMPLEX) ? 2 : 1;

    vector_t *vCopia = vector_new(v->len, C->type, 1, NULL);
    vector_t *v_B = vector_new(v->len, C->type, 1, NULL);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < v->len; i++) {
        vCopia->value.f[aux * i] = v->value.f[aux * i];
        if (C->type == T_COMPLEX) vCopia->value.f[aux * i + 1] = v->value.f[aux * i + 1];
    }

    sparse_matrix_t S_mkl, C_mkl, U_mkl;
    struct matrix_descr descr = {.type = SPARSE_MATRIX_TYPE_GENERAL};

    // Conversões
    mkl_sparse_d_create_csr(&S_mkl, SPARSE_INDEX_BASE_ZERO, S->nrow, S->ncol,
                            S->row_ptr, S->row_ptr + 1, S->col_idx, S->values);
    mkl_sparse_d_create_csr(&C_mkl, SPARSE_INDEX_BASE_ZERO, C->nrow, C->ncol,
                            C->row_ptr, C->row_ptr + 1, C->col_idx, C->values);

    // Repetições principais
    for (int g = 1; g <= REPETICOESG; g++) {
        if (g == 1 || g == REPETICOESG)
            printf("++ ++ %d from %d REPETICOESG\n", g, REPETICOESG);

        clock_gettime(CLOCK_MONOTONIC, &startP);
        for (int r = 1; r <= REPETICOES; r++) {
            if (r == 1 || r == REPETICOES)
                printf(" repeticao: %d/%d\n", r, REPETICOES);

            mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, S_mkl, C_mkl, &U_mkl);
        }
        clock_gettime(CLOCK_MONOTONIC, &endP);
        tempo_decorridoP += (endP.tv_sec - startP.tv_sec) + (endP.tv_nsec - startP.tv_nsec) / 1e9;

         int status = mkl_sparse_optimize(U_mkl);
         if (status != SPARSE_STATUS_SUCCESS) {
             printf("Erro em mkl_sparse_optimize\n");
             exit( 2222);
         }

        clock_gettime(CLOCK_MONOTONIC, &startI);
        for (int r = 1; r <= REPETICOES; r++) {
            if (r == 1 || r == REPETICOES)
                printf(" repeticao: %d/%d\n", r, REPETICOES);

            // Reinicializa vetores
            #pragma omp parallel for
            for (int i = 0; i < v_B->len; i++)
                v_B->value.f[aux * i] = 0.0;

            #pragma omp parallel for
            for (int i = 0; i < v->len; i++)
                v->value.f[aux * i] = vCopia->value.f[aux * i];

            vector_t *v_input = v, *v_output = v_B;

            for (int i = 1; i <= numIterations; i++) {
                if (g == 1 || g == REPETICOESG)
                    if (r == 1 || r == REPETICOES)
                        printf(" iteracao %d/%d %d/%d\n", i, numIterations, r, REPETICOES);

                if (i % 2 == 1) {
                    v_input = v;
                    v_output = v_B;
                } else {
                    v_input = v_B;
                    v_output = v;
                }


                // Multiplicação U * v_input usando MKL
                mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                                1.0, U_mkl, descr, v_input->value.f,
                                0.0, v_output->value.f);

                if (g == 1 || g == REPETICOESG)
                    if (r == 1 || r == REPETICOES) {
                        mostrarConteudoV(v_input,  "  em simulateMKL,  v_input");
                        mostrarConteudoV(v_output, "  em simulateMKL, v_output");
                    }
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &endI);
        tempo_decorridoI += (endI.tv_sec - startI.tv_sec) + (endI.tv_nsec - startI.tv_nsec) / 1e9;
    }

    printf("Tempo de %d permutacoes   : %.6f s\n", REPETICOESG * REPETICOES, tempo_decorridoP);
    printf("Tempo medio por permutacao: %.6f s\n", tempo_decorridoP / (REPETICOESG * REPETICOES));
    printf("Tempo de %d iteracoes     : %.6f s\n", REPETICOESG * REPETICOES * numIterations, tempo_decorridoI);
    printf("Tempo medio por iteracao  : %.6f s\n", tempo_decorridoI / (REPETICOESG * REPETICOES * numIterations));
         #pragma omp parallel for schedule(guided)
         for (int i = 0; i < v_B->len; i++){
                                 v_B->value.f[aux*i]  =0.;
          if(C->type==T_COMPLEX) v_B->value.f[aux*i+1]=0.;
         }

         #pragma omp parallel for schedule(guided)
         for (int i = 0; i < v->len; i++) {
                                      v->value.f[aux*i]  =vCopia->value.f[aux*i];
              if(C->type==T_COMPLEX) v->value.f[aux*i+1]=vCopia->value.f[aux*i+1];
	 }



    // Cleanup MKL handles
    mkl_sparse_destroy(S_mkl);
    mkl_sparse_destroy(C_mkl);
    mkl_sparse_destroy(U_mkl);
    #endif
}


void simulateLibC(smatrix_t *S, smatrix_t *C, smatrix_t *U_, vector_t *v){
    struct timespec startP, endP; double tempo_decorridoP=0.0;
    struct timespec startI, endI; double tempo_decorridoI=0.0 ;

    printf("em simulateLibC ...\n");
//    labelComputeU="??????????\n";

    int aux=1; if(C->type==T_COMPLEX) aux =2;

    vector_t * v_outputTmp=NULL;
    smatrix_t * U=NULL ;

    vector_t * v_input, *v_output;
    vector_t * vCopia = vector_new(v->len, C->type, 1, NULL );
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < v->len; i++) {
                              vCopia->value.f[aux*i]  =v->value.f[aux*i];
       if(C->type==T_COMPLEX) vCopia->value.f[aux*i+1]=v->value.f[aux*i+1];
    }

    vector_t * v_B = vector_new(v->len, C->type, 1, NULL );
    U = smatrix_new(S->nrow, S->nrow, C->type);


    allocate_result(S, C, U);
int r, g;


tempo_decorridoP=0.0;
tempo_decorridoI=0.0;
for(g = 1; g <= REPETICOESG; g++){

    if ( g==1 || g==REPETICOESG) printf("++ ++ %d from %d REPETICOESG\n", g, REPETICOESG); // exit(0);

    clock_gettime(CLOCK_MONOTONIC, &startP);
    for( r = 1; r <= REPETICOES; r++){
       if ( r == 1 || r == REPETICOES ) printf(" repeticao: %d/%d  \n", r,REPETICOES);
       if(computeU){
          // U = S * C; 
           labelComputeU="algebra via  libC: permute_and_multiply; smatrix_vector_mult, computeU = 1; true \n";
           compute_U_row_ptr(S, C, U);
           permute_and_multiply(S, C, U);
       } else{
           labelComputeU="v_tmp = C x v_in; v_out = S x v_tmp; computeU = 0; false \n";
           U=C;
           vector_delete(v_outputTmp);
           v_outputTmp = vector_new(v->len, C->type, 1, NULL );
       }
         if(g==1 || g==REPETICOESG) if(r==1 || r==REPETICOES)
		 if(U->nrow<25) { printf("\n U = \n"); print_dense(U); print_smatrix(U); }

    } // for(int r = 1; r <= REPETICOES; r++){
    clock_gettime(CLOCK_MONOTONIC, &endP);
    tempo_decorridoP += (endP.tv_sec - startP.tv_sec) + (endP.tv_nsec - startP.tv_nsec)/ 1e9;
     
      clock_gettime(CLOCK_MONOTONIC, &startI);

      for(int r = 1; r <= REPETICOES; r++){
         if ( r == 1 || r == REPETICOES ) printf(" repeticao: %d/%d  \n", r,REPETICOES);

         #pragma omp parallel for schedule(guided)
         for (int i = 0; i < v_B->len; i++){
                                 v_B->value.f[aux*i]  =0.;
          if(C->type==T_COMPLEX) v_B->value.f[aux*i+1]=0.;
         }

         #pragma omp parallel for schedule(guided)
         for (int i = 0; i < v->len; i++) {
                                      v->value.f[aux*i]  =vCopia->value.f[aux*i];
              if(C->type==T_COMPLEX) v->value.f[aux*i+1]=vCopia->value.f[aux*i+1];
	 }

         v_input=v; v_output = v_B;

         for(int i = 1; i <= numIterations; i++){

            if(g==1 || g==REPETICOESG) 
               if ( r == 1 || r == REPETICOES ) printf(" iteracao %d/%d  %d/%d - de %d \n", i,numIterations,r,REPETICOES, REPETICOESG);

            if(i%2==1){ // odd
                 v_input=v; v_output = v_B; }
            else     { //even
                 v_input=v_B; v_output = v; }

            if(computeU) {
               smatrix_vector_mult   (U, v_input, v_output);
            } else{
               smatrix_vector_mult   (U, v_input,     v_outputTmp);
               permute_vector        (S, v_outputTmp, v_output); 
            }

            if(g==1 || g==REPETICOESG) if(r==1 || r==REPETICOES){
                mostrarConteudoV(v_input,  " em simulateLibC,  v_input");
                mostrarConteudoV(v_output, " em simulateLibC, v_output");
           }
      } // for(int i = 1; i <= numIterations; i++){

    } // for(int r = 1; r <= REPETICOES; r++){
     

      clock_gettime(CLOCK_MONOTONIC, &endI);
      tempo_decorridoI += (endI.tv_sec - startI.tv_sec) + (endI.tv_nsec - startI.tv_nsec) / 1e9;
//  if(!computeU) vector_delete(v_outputTmp);

  } // for(int g = 1; g <= 10*REPETICOESG; g++){
     
    printf("Tempo de %d permutacoes   : %.6f s\n", (REPETICOESG)*REPETICOES,tempo_decorridoP);
    printf("Tempo medio por permutacao: %.6f s\n", tempo_decorridoP/(REPETICOESG*REPETICOES));
    printf("Tempo de %d iteracoes     : %.6f s\n", (REPETICOESG)*REPETICOES*numIterations,  tempo_decorridoI);
    printf("Tempo medio por iteracao  : %.6f s\n", tempo_decorridoI / (REPETICOESG * REPETICOES * numIterations));

    double densidade=(1.0*U->nnz)/((1L * U->nrow) * U->nrow) * 100;
    printf("arquivo de entrada: %s\n", arqEntrada);
    printf("N = %d, nnz = %d, densidade = %7.4f %%, computeU = %s \n", C->nrow, C->nnz, densidade,  computeU ? "true" : "false" );
    printf("%s",labelComputeU);

         #pragma omp parallel for schedule(guided)
         for (int i = 0; i < v_B->len; i++){
                                 v_B->value.f[aux*i]  =0.;
          if(C->type==T_COMPLEX) v_B->value.f[aux*i+1]=0.;
         }

         #pragma omp parallel for schedule(guided)
         for (int i = 0; i < v->len; i++) {
                                      v->value.f[aux*i]  =vCopia->value.f[aux*i];
              if(C->type==T_COMPLEX) v->value.f[aux*i+1]=vCopia->value.f[aux*i+1];
	 }

//exit(-3);
}

void print_vector(vector_t *v){
   printf("Vector: ");
    for (int i = 0; i < v->len; i++) printf("%.2f ", v->value.f[i]);
    printf("\n");
}

void print_v(float *v, int n){
    for (int i = 0; i < n; i++) printf("%.2f ", v[i]);
}


void print_vectorR(const vector_t *v, int inicio_, int fim_){
    if(fim_==inicio_) return;
    setvbuf(stdout, NULL, _IONBF, 0);
    double sum=0.0;
    int i;
    for (i = 0; i < v->len; i++){
       sum+=v->value.f[i]*v->value.f[i];
       if(i==inicio_) printf("[%d:%d]:", inicio_, fim_);
       if(i >= inicio_ &&  i <= fim_) { printf(" %11.4e", v->value.f[i]); }
    }
    if(fim_ == v->len-1) printf("\nL2 norm = %f\n", sqrt(sum));
    printf("\n");
}

void print_vectorC(const vector_t *v, int inicio_, int fim_) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if(fim_ < inicio_ || inicio_ >= v->len) return;  // Verificação segura de limites

    // Ajustar fim_ se necessário
    if(fim_ >= v->len) fim_ = v->len - 1;

    double sum_sq = 0.0;  // Acumulador para soma dos quadrados
    int header_printed = 0;  // Controle de impressão do cabeçalho

    // Loop por TODOS os elementos para calcular a norma corretamente
    for (int i = 0; i < v->len; i++) {
        double a = v->value.f[2*i];
        double b = v->value.f[2*i+1];
        
        // Cálculo CORRETO do módulo quadrado: |z|² = a² + b²
        sum_sq += a*a + b*b;

        // Imprimir cabeçalho na primeira iteração dentro do intervalo
        if(!header_printed && i >= inicio_) {
            //printf("Elementos [%d-%d]: ", inicio_, fim_);
            printf("[%d-%d]: ", inicio_, fim_);
            header_printed = 1;
        }
        
        // Imprimir elementos no intervalo solicitado
        if(i >= inicio_ && i <= fim_) {
            printC(a, b);
            printf(", ");
        }
    }

    // Calcular e imprimir norma L2 (valor real) quando solicitado
    if(fim_ == v->len-1) {
        double norm = sqrt(sum_sq);  // Norma = raiz quadrada da soma
        printf("\nL2 norm = %.4e", norm);  // Formato científico
    }
    printf("\n");
}

/*
void print_vectorC(const vector_t *v, int inicio_, int fim_){
    setvbuf(stdout, NULL, _IONBF, 0);
    if(fim_==inicio_) return;
    setvbuf(stdout, NULL, _IONBF, 0);
    double aSum=0.0, bSum=0;
    int i;
    for (i = 0; i < v->len; i++){
       double a=v->value.f[2*i];
       double b=v->value.f[2*i+1];
       double aP=a * a - b * b, bP=2 * a * b;
       aP=a * a + b * b ;  bP=0.0;
       aSum+=aP; bSum+=bP;
       if(i==inicio_) printf("..%d:%d..:", inicio_, fim_);
       if(i >= inicio_ &&  i <= fim_) { printC(a,b); printf(", "); }
    }
    if(fim_ == v->len-1) {printf("\nL2 norm = "); printC(aSum, bSum);}
    printf("\n");
}
*/

void mostrarConteudoV(vector_t *v, char* label_) {
{     
    int inicio = 0, fim = inicio+4;
    if (  v->len <=50 ) fim = v->len-1;
            printf("%s", label_);
            //print_tVector(v, inicio, fim);
            //printf(v->type == T_FLOAT ? "Vector R: " : "Vector C: ");
            if(v->type == T_FLOAT)
                print_vectorR(v, inicio, fim);
            else
                print_vectorC(v, inicio, fim);

    if (  v->len > 50 ){
        inicio = v->len-5; fim = inicio+4;
            printf("   o o o   "); 
            //print_tVector(v, inicio, fim);
            if(v->type == T_FLOAT)
                print_vectorR(v, inicio, fim);
            else
                print_vectorC(v, inicio, fim);
        }
    }
}

void print_smatrix(const smatrix_t* matrix) {
    if (!matrix) {
        printf("Matrix is NULL.\n");
        return;
    }

    printf("Matrix (%p):\n", (void*)matrix);
    printf("  Rows     : %d, Cols   : %d, NNZ    : %d\n", matrix->nrow, matrix->ncol, matrix->nnz);
    printf("  isPacked : %d\n", matrix->isPacked);
    printf("  type     : %d\n", matrix->type);
    printf("  location : %u\n", matrix->location);
    printf("  extra    : %p\n", matrix->extra);
    printf("  idxColMem: %p\n", matrix->idxColMem);
    int limite=30;

    if (matrix->row_ptr) {
        printf("  row_ptr: ");
int n = matrix->nrow;
int inicio=0, fim=matrix->nrow;
if ( matrix->nrow > limite) { fim = limite; }
        for (int i = inicio; i <= fim; i++) { printf("%d ", matrix->row_ptr[i]); }
if ( matrix->nrow > limite){ printf("   o o o   ");
inicio=matrix->nrow - limite, fim= matrix->nrow;
        for (int i = inicio; i <= fim; i++) { printf("%d ", matrix->row_ptr[i]); }
}
        printf("\n");
    } else {
        printf("  row_ptr is NULL.\n");
    }

    if (matrix->col_idx) {
        printf("  col_idx: ");
int n = matrix->nnz;
int inicio=0, fim= matrix->nnz;
if ( matrix->nnz > limite) { fim = limite; }
        for (int i = inicio; i < fim; i++) { printf("%d ", matrix->col_idx[i]); }
if ( matrix->nnz > limite){ printf("   o o o   ");
inicio=matrix->nnz-limite, fim= matrix->nnz;
        for (int i = inicio; i < fim; i++) { printf("%d ", matrix->col_idx[i]); }}
        printf("\n");
    } else {
        printf("  col_idx is NULL.\n");
    }

    void (*printElemntP) (const double *) = printRp;
    
    if(matrix->type == T_FLOAT)
        printElemntP = printRp;    
    else
        printElemntP = printCp;

    int i; int aux=1;
    if(matrix->type==T_COMPLEX) aux=2;
    if (matrix->values) {
        printf("  values: ");
	int n = matrix->nnz;
	int inicio=0, fim= matrix->nnz;
	if ( matrix->nnz > limite) { fim = limite; }
		//for (i = inicio; i < fim; i++) { printf("%.4f ", matrix->values[i]); }
	for (i = inicio; i < fim; i++) { printElemntP(&matrix->values[aux*i]); } 
	if ( matrix->nnz > limite) { printf("   o o o   ");
	inicio=matrix->nnz-limite, fim= matrix->nnz;
		//for (int i = inicio; i < fim; i++) { printf("%.4f ", matrix->values[i]); }}
	for (i = inicio; i < fim; i++) { printElemntP(&matrix->values[aux*i]); }  }
		printf("\n");
    } else {
        printf("  values is NULL.\n");
    }
    return;
}

void print_dense(const smatrix_t* mat) {
    printf(" em print_dense, nrow =%d\n", mat->nrow );
    //for (int i = 0; i < mat->nrow+1; i++) { printf(".. %d, ", mat->row_ptr[0]); }
    //printf(" \n");

    int aux = 1;
    void (*printElemntP) (const double *) = printRp;
    if( mat->type == T_COMPLEX) {
            printElemntP  = printCp;
            aux = 2;
    }

    for (int i = 0; i < mat->nrow; i++) {
        int row_start = mat->row_ptr[i];
        int row_end   = mat->row_ptr[i + 1];
        for (int j = 0; j < mat->ncol; j++) {
            int found = 0;
            for (int k = row_start; k < row_end; k++) {
                if (mat->col_idx[k] == j) {
                    //printf(" %5.2f", mat->values[k]);
                    printElemntP(&mat->values[k*aux]); printf(" ");
                    found = 1;
                    break;
                }
            }
            if (!found) {
                //printf(" %5.2f", 0.0);
                double zero[]={0.0, 0.0};
                printElemntP(zero);  printf(" ");
            }
        }
        printf("\n");
    }
return;
}


// Gera uma matriz de permutação aleatória N×N
void generate_Permutation_matrix(smatrix_t* matP_) {
    printf("em void generate_Permutation_matrix(smatrix_t* matP, ...\n");
    int* perm = malloc(matP_->nrow * sizeof(int));
    if (!perm) return;

    // Inicializa a permutação como identidade
    for (int i = 0; i < matP_->nrow; i++) { perm[i] = i; }

    // Embaralha usando Fisher-Yates
    srand(1000); //srand(time(NULL));
    for (int i = matP_->nrow-1 ; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = perm[i]; perm[i] = perm[j]; perm[j] = temp;
    }
     //printf("perm ="); for (int i = matP_->nrow - 1; i >= 0; i--) { printf("%d, ", perm[i]); } printf("\n");

    int block_size=1;
    matP_->row_ptr = (int *)    malloc((matP_->nrow+1) * sizeof(int));
    matP_->row_ptr[0] = 0;
    int NNZ=matP_->nrow;
    matP_->col_idx = (int *)    malloc(NNZ * sizeof(int));
    matP_->values  = (double *) malloc(NNZ * sizeof(double));
    matP_->nnz=0;
    matP_->isPacked = 1;

    // Define os 1s na matriz de permutação
    for (int i = 0; i < matP_->nrow; i++) {
        matP_->col_idx[matP_->nnz] = perm[i];
        matP_->values[matP_->nnz]  = 1.0;
        matP_->nnz++;
        matP_->row_ptr[i+1]    = matP_->nnz;
    }
    free(perm);
}

void generate_block_diagonal_matrix(smatrix_t* matC, int block_size_, float variation_) {
    printf("em void generate_block_diagonal_matrix(smatrix_t* matC, ...\n");
    int nnz = 0, row = 0, col;
    long nunTotElem=(long ) matC->nrow*matC->nrow;

    printf(  "\n");
    printf(  "Matrix %d X %d  \n", matC->nrow,matC->nrow);
    printf(  "num total elem = %ld\n", nunTotElem);
    printf(  "num total elem = %ld\n", nunTotElem);

    matC->row_ptr = (int *)    malloc((matC->nrow+1) * sizeof(int));
    matC->row_ptr[0] = 0;
    matC->col_idx = (int *)    malloc(matC->nrow*block_size_*1.2 * sizeof(int));
    matC->values  = (double *) malloc(matC->nrow*block_size_*1.2 * sizeof(double));
    matC->nnz=0;
    matC->isPacked = 1;

    int block_size ;
    while (row < matC->nrow) {
        block_size = block_size_ * (1.0 + variation_ * ((rand() % 201 - 100) / 100.0));
        if (row + block_size > matC->nrow){ block_size = matC->nrow - row; }
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j <= block_size-1; j++) {
                col = row + j;
double valor = 1.0/block_size;
                matC->col_idx[matC->nnz] = col;
                matC->values[matC->nnz] = valor;
                matC->nnz++;
            }
            matC->row_ptr[row+i+1] = matC->nnz;
        }
        nnz += block_size*block_size;
        row += block_size;
    }
    printf(  "num nonzeros   = %d, %d\n", nnz, (int)(matC->nrow*block_size_*1.2));
    printf(  "densidade      = %f %%\n", (1.0*nnz)/(nunTotElem) * 100);
    printf(  "sizeof(int)    = %ld \n", sizeof(int));
    printf(  "sizeof(double) = %ld \n", sizeof(double));
    float byteAmountSMat= (sizeof(int)*2+sizeof(double))*(nnz)/(1000*1000) ;
    printf(  "Memory estimate  = %f Mbyte\n", byteAmountSMat );
   return;
}


int N, BLOCK_SIZE; float VARIATION;
void setSMatrix(smatrix_t **matS_, int *perm, int N, data_type tipo) {
    *matS_ = smatrix_new(N, N, tipo);
    smatrix_t *mat = *matS_;

    mat->row_ptr = (int *) malloc((N + 1) * sizeof(int));
    mat->col_idx = (int *) malloc(N * sizeof(int));
    mat->nnz = N;
    mat->nrow = mat->ncol = N;
    mat->isPacked = 1;
    mat->type = tipo;

    if (tipo == T_COMPLEX) {
        mat->values = (double *) malloc(2 * N * sizeof(double)); // 2 valores por entrada (real, imag)
    } else {
        mat->values = (double *) malloc(N * sizeof(double));
    }

    mat->row_ptr[0] = 0;
    for (int i = 0; i < N; i++) {
        mat->col_idx[i] = perm[i];
        if (tipo == T_COMPLEX) {
            mat->values[2 * i]     = 1.0; // parte real
            mat->values[2 * i + 1] = 0.0; // parte imaginária
        } else {
            mat->values[i] = 1.0;
        }
        mat->row_ptr[i + 1] = i + 1;
    }
}

void includeSCMatrices(smatrix_t ** matC_, smatrix_t ** matS_, vector_t ** vI_){
{
  //  printf("valores default, abril de 2025, por Gustavo\n");
   int N=8;
   int nnz=20;
   numIterations=9;
   arqEntrada="entradasCoined_04_25/177R.txt";
    int perm[]={ 4,5,6,7,0,1,2,3 };

    *matS_ = smatrix_new(N, N, T_FLOAT);
    (*matS_)->row_ptr = (int *)    malloc(((*matS_)->nrow+1) * sizeof(int));
    (*matS_)->row_ptr[0] = 0;
    (*matS_)->nnz=N;
    (*matS_)->col_idx = (int *)    malloc((*matS_)->nnz * sizeof(int));
    (*matS_)->values  = (double *) malloc((*matS_)->nnz * sizeof(double));
    (*matS_)->isPacked = 1;

    // Define os 1s na matriz de permutação
    for (int i = 0; i < (*matS_)->nrow; i++) {
        (*matS_)->col_idx[i]    = perm[i];
        (*matS_)->values[i]     = 1.0;
        (*matS_)->row_ptr[i+1]  = i+1;
    }
    //if((*matS_)->nrow<25) { printf("\n matS = \n"); print_dense((*matS_)); print_smatrix((*matS_)); }

int row_ptr[]={ 0,4,8,12,16,17,18,19,20 };
int col_idx[]={ 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,4,5,6,7 };
double values[]={ 0.5,-0.5,-0.5,-0.5,-0.5,0.5,-0.5,-0.5,-0.5,-0.5,0.5,-0.5,-0.5,-0.5,-0.5,0.5,-1.0,-1.0,-1.0,-1.0 };

    *matC_ = smatrix_new(N, N, T_FLOAT);

    (*matC_)->row_ptr = (int *)    malloc(((*matC_)->nrow+1) * sizeof(int));
    (*matC_)->row_ptr[0] = 0;
    (*matC_)->nnz=nnz;
    (*matC_)->col_idx = (int *)    malloc((*matC_)->nnz * sizeof(int));
    (*matC_)->values  = (double *) malloc((*matC_)->nnz * sizeof(double));
    (*matC_)->isPacked = 1;

    for (int i = 0; i <=  (*matC_)->nrow; i++) {(*matC_)->row_ptr[i] = row_ptr[i];}
    for (int i = 0; i <    (*matC_)->nnz; i++) {(*matC_)->col_idx[i] = col_idx[i];};
    for (int i = 0; i <    (*matC_)->nnz; i++) {(*matC_)->values[i]  = values[i];}; 

    //if((*matC_)->nrow<25) { printf("\n matC_ = \n"); print_dense((*matC_)); print_smatrix((*matC_)); }

       double vInicial[]={ 1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
*vI_ = vector_new(N, T_FLOAT, 1, NULL ); for (int i = 0; i < (*vI_)->len; i++) {(*vI_)->value.f[i] = vInicial[i];}
}

#ifdef INPUTDATA
    #include "inputSCMatricesC" // arquivo criado pelo shell script inputData.sh
#endif

} // void includeSCMatrices(smatrix_t ** matC_,

void createSCMatrices(smatrix_t ** matC_, smatrix_t ** matS_, vector_t ** vI_){
    DadosCoined *dados;
    dados = (DadosCoined*) malloc (sizeof(DadosCoined));

    N=6000; BLOCK_SIZE=400; VARIATION=0.0;
    N=10;      BLOCK_SIZE=5;   VARIATION=0.0;
    N=600000; BLOCK_SIZE=400; VARIATION=0.0;

    int SEMENTE=1; srand(SEMENTE);

    *matS_ = smatrix_new(N, N, T_FLOAT);
    *matC_ = smatrix_new(N, N, T_FLOAT);
    *vI_   =  vector_new(N,    T_FLOAT, 1, NULL );
         for (int i = 0; i < (*vI_)->len; i++) (*vI_)->value.f[i] = 1./1.;

    dados->perm=NULL;
    dados->N=N;
    dados->sMat=**matC_;
    dados->vInicial=(*vI_)->value.f;
    dados->numIteracoes=2;
    dados->iteracoes=NULL;

    generate_Permutation_matrix   (*matS_);

    generate_block_diagonal_matrix(*matC_, BLOCK_SIZE, VARIATION);
    return ;
} //void createSCMatrices(smatrix_t ** matC_, smatrix_t ** matS_, vector_t ** vI_){

void setCoinedData(smatrix_t ** matC_, smatrix_t ** matS_, vector_t ** vI_){

/*
    createSCMatrices  (matC_, matS_, vI_);
    includeSCMatrices (matC_, matS_, vI_);
*/

    arqEntrada = "entradasCoined_05_25/177R.txt";
    arqEntrada = "entradasCoined_05_25/30C.txt";
    REPETICOES = 1; REPETICOESG=2;
    computeU   = 0;  // 0 é false, permuta VETOR, U = C, v_t=U x v_input, v_out =U x v_t
//
    printf( " em setCoinedData \n");
    if (ARGC != 5) {
        fprintf(stderr, "Uso: %s <arquivo.txt> <REPETICOES> <computeU> <REPETICOESG>\n", ARGV[0]);
        fprintf(stderr, "Uso: %s %s %d %d %d\n", ARGV[0], arqEntrada, REPETICOES, computeU, REPETICOESG );
        exit(1); }

    arqEntrada =      ARGV[1];
    REPETICOES = atoi(ARGV[2]);
    computeU   = atoi(ARGV[3]); // 0 é false, U = C,  permuta V_out, v_t=U x v_input, v_out =U x v_t
                                // 1 é true,  computa U = S x C; v_out=U x v_input
    REPETICOESG = atoi(ARGV[4]);
    int tipo;

    DadosCoined *dados;

    if        (strstr(arqEntrada, "R-")) {
        printf(" em setCoinedData  call lerDadosCoinedR\n");
                dados = lerDadosCoinedR(arqEntrada);
        print_tVector = print_vectorR;
        printElemntP  = printRp;
        tipo          = T_FLOAT;
    } else if (strstr(arqEntrada, "C-")) {
        printf(" em setCoinedData  call lerDadosCoinedC\n");
                dados = lerDadosCoinedC(arqEntrada);
        print_tVector = print_vectorC;
        printElemntP  = printCp;
        tipo          = T_COMPLEX;
    } else {
        fprintf(stderr, "%s\n Erro: nome de arquivo deve terminar com 'R.txt' ou 'C.txt'\n", arqEntrada);
        return ;
    }

    if (!dados) { fprintf(stderr, "Erro ao ler o arquivo: %s\n", arqEntrada); return ; }

    //printf("arquivo de entrada: %s\n", arqEntrada);
    printf("Iterações no arquivo = %d\n", dados->numIteracoes );
    //printf("N = %d, nnz = %d, REPETICOES = %d, computeU = %s \n", dados->N, dados->sMat.nnz, REPETICOES, computeU ? "true" : "false" );

    *matC_=&dados->sMat;
    (*matC_)->type=tipo;

    setSMatrix(matS_, dados->perm, dados->N, (*matC_)->type);

    *vI_= vector_new(dados->N, (*matC_)->type, 1, NULL );
    (*vI_)->value.f=dados->vInicial;
    mostrarConteudoV(*vI_, "em setCoinedData,  vInLido");
  //   printf("aqui oh!\n"); exit(-2);
   
    numIterations=dados->numIteracoes;
    vector_t  *vOutLido = vector_new(dados->N, (*matC_)->type, 1, NULL );
//*
    int i;
    for (i = 0; i <  dados->numIteracoes-1; i++) {
        double aSum=0.0, bSum=0.0; int k = 0;
        printf("Iteração %d:\n", i+1);
        vOutLido->value.f=dados->iteracoes[i];
        mostrarConteudoV(vOutLido, "em setCoinedData, vOutLido");
     } // for (i = 0; i <  dados->numIteracoes; i++) {
     printf("Iteração %d:\n", i+1);
     vOutLido->value.f=dados->iteracoes[i];
     mostrarConteudoV(vOutLido, "Vetor output final      Lido");
//  */
   
    // free_dadosCoined(dados);
   }


int main(int argc, char *argv[]) {
    setbuf(stdout, NULL);
    ARGC=argc; ARGV=argv;

    smatrix_t *S, *C;
    vector_t *v;

    setCoinedData   (&C, &S, &v);
    if(S->nrow<25) { printf("\n S = \n");     print_dense(S); print_smatrix(S); }
    if(C->nrow<25) { printf("\n C = \n");     print_dense(C); print_smatrix(C); }
   
    smatrix_t * U = smatrix_new(S->nrow, S->nrow, C->type);
    allocate_result(S, C, U);
   
    int numMaxThrs=omp_get_max_threads();
    double densidade=(1.0*C->nnz)/((1L * C->nrow) * C->nrow) * 100;
    printf("arquivo de entrada: %s\n", arqEntrada);
    
    /*

    labelComputeU="algebra via  libC: permute_and_multiply; smatrix_vector_mult, computeU = 1; true \n";
    printf("numMaxThrs= %d, N= %d, nnz= %d, densidade= %6.4f %%, exe,",numMaxThrs, C->nrow, C->nnz, densidade );
    printf(" %s", labelComputeU);
    printf(" em main: simulateLibC(S, C, U, v);\n");
    simulateLibC(S, C, U, v);
    */
    
    labelComputeU="algebra via   MKL: U=mkl_sparse_spmm(S_, C_); y=mkl_sparse_spmv(U, x), computeU = 1; true\n";
    printf("numMaxThrs= %d, N= %d, nnz= %d, densidade= %6.4f %%, exe,",numMaxThrs, C->nrow, C->nnz, densidade );
    printf(" %s", labelComputeU);
    printf(" em main: simulateMKL (S, C, U, v);\n");
    simulateMKL (S, C, U, v);


    return 0;

    smatrix_delete(C);
    smatrix_delete(S);
    smatrix_delete(U);
    return 0;
}
