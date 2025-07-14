#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

typedef union __data_vector_u {
    void               * v;
    int                * i;
    double             * f;
    void              ** s;
} data_vector_u;

typedef struct __vector_t {
    data_vector_u      value;
    int                len;
    data_type          type;
    unsigned char      location;
    void*              extra;
    int                externalData;
} vector_t;

typedef struct {
    int *perm;
    int N;
    smatrix_t sMat;
    double *vInicial;
    int numIteracoes;
    double **iteracoes;
} DadosCoined;

void print_smatrix(smatrix_t * );

// Lê inteiros de uma linha e armazena dinamicamente

int *ler_inteiros(char *linha, int *out_count) {
    int capacidade = 1024;
    int count = 0;
    int *valores = malloc(capacidade * sizeof(int));
    if (!valores) return NULL;

    char *ptr = linha;
    while (*ptr) {
        while (*ptr && isspace(*ptr)) ptr++;  // pula espaços

        char *endptr;
        int valor = strtol(ptr, &endptr, 10);

        if (ptr == endptr) break;  // não conseguiu converter

        if (count >= capacidade) {
            capacidade *= 2;
            int *tmp = realloc(valores, capacidade * sizeof(int));
            if (!tmp) {
                free(valores);
                return NULL;
            }
            valores = tmp;
        }

        valores[count++] = valor;
        ptr = endptr;
    }

    *out_count = count;
    return valores;
}

// Lê doubles de uma linha e armazena dinamicamente
double *ler_doubles(char *linha, int *out_count) {
    int capacidade = 1024;
    int count = 0;
    double *valores = malloc(capacidade * sizeof(double));
    if (!valores) return NULL;
    char *ptr = linha;
    while (*ptr) {
        while (isspace(*ptr)) ptr++;  // pula espaços
        char *endptr;
        double valor = strtod(ptr, &endptr);
        if (ptr == endptr) break;  // nada foi lido
        if (count >= capacidade) {
            capacidade *= 2;
            double *tmp = realloc(valores, capacidade * sizeof(double));
            if (!tmp) { free(valores); return NULL; }
            valores = tmp;
        }
        valores[count++] = valor;
        ptr = endptr;
    }
    *out_count = count;
    return valores;
}

// Função para ler valores complexos no formato (a±bj), bj ou -bj
// e armazenar como vetor intercalado de doubles: [real0, imag0, real1, imag1, ...]
double *ler_complexos_como_doubles(char *linha, int *out_count) {
    int capacidade = 1024;
    int count = 0;
    double *valores = malloc(capacidade * sizeof(double));
    if (!valores) return NULL;

    char *ptr = linha;
    while (*ptr) {
        while (isspace(*ptr)) ptr++;

        // Caso especial: número imaginário puro (ex: 0j, -3j)
        if (*ptr == '-' || isdigit(*ptr)) {
            char *endptr;
            double imag = strtod(ptr, &endptr);
            ptr = endptr;
            while (isspace(*ptr)) ptr++;

            if (*ptr == 'j' || *ptr == 'J') {
                ptr++;  // pula 'j'

                if (count + 2 > capacidade) {
                    capacidade *= 2;
                    double *tmp = realloc(valores, capacidade * sizeof(double));
                    if (!tmp) { free(valores); return NULL; }
                    valores = tmp;
                }

                valores[count++] = 0.0;   // real
                valores[count++] = imag;  // imag
                //printf("real = %g, imag = %g (imaginário puro)\n", 0.0, imag);
                continue;
            } else {
                ptr = endptr; // retrocede, não era um complexo válido
            }
        }

        if (*ptr != '(') break; // exige parêntese para o formato completo
        ptr++;  // pula '('

        char *endptr;
        double real = strtod(ptr, &endptr);
        if (ptr == endptr) break;
        ptr = endptr;

        while (isspace(*ptr)) ptr++;

        int negativo = 0;
        if (*ptr == '+' || *ptr == '-') {
            negativo = (*ptr == '-') ? 1 : 0;
            ptr++;
        } else break;

        double imag = strtod(ptr, &endptr);
        if (ptr == endptr) break;
        ptr = endptr;

        while (isspace(*ptr)) ptr++;
        if (*ptr != 'j' && *ptr != 'J') break;
        ptr++;

        while (isspace(*ptr)) ptr++;
        if (*ptr != ')') break;
        ptr++;

        if (count + 2 > capacidade) {
            capacidade *= 2;
            double *tmp = realloc(valores, capacidade * sizeof(double));
            if (!tmp) { free(valores); return NULL; }
            valores = tmp;
        }

        valores[count++] = real;
        valores[count++] = negativo ? -imag : imag;

        //printf("real = %g, imag = %g\n", real, negativo ? -imag : imag);
    }

    *out_count = count / 2;
    return valores;
}

DadosCoined*  lerDadosCoinedR(const char *fileName) {
    printf(" em DadosCoined*  lerDadosCoinedR(const char *fileName) {\n");

    printf("arquivo de entrada: %s\n", fileName);
    FILE *f = fopen(fileName, "r");
    if (!f) { fprintf(stderr, "Erro ao abrir o arquivo: %s\n", fileName); return NULL; }

    DadosCoined *dados = malloc(sizeof(DadosCoined));
    dados->sMat.type = T_FLOAT;

    int SIZEOFLINHA=2048*2048*100;
    char *linha = (char*) malloc (SIZEOFLINHA);

    // 1ª linha - perm
    printf("ler perm\n");
    char *l  = fgets(linha, SIZEOFLINHA, f);
    dados->perm = ler_inteiros(linha, &dados->N);
    int N = dados->N;

    printf("N=%d\n",dados->N);

    // 2ª linha - row_ptr
    printf("ler row_ptr\n");
    int n_row_ptr;
    l = fgets(linha, SIZEOFLINHA, f);
    dados->sMat.row_ptr = ler_inteiros(linha, &n_row_ptr);
    dados->sMat.nrow= n_row_ptr-1;
    dados->sMat.ncol= n_row_ptr-1;
    dados->sMat.nnz = dados->sMat.row_ptr[n_row_ptr - 1];
   
    // 3ª linha - col_idx
    printf("ler col_idx\n");
    int count;
    l = fgets(linha, SIZEOFLINHA, f);
    dados->sMat.col_idx = ler_inteiros(linha, &count);
    if (count != dados->sMat.nnz) {
        fprintf(stderr, "Erro: col_idx com tamanho %d, esperado %d\n", count, dados->sMat.nnz);
        fclose(f); return NULL; }

    // 4ª linha - values
    printf("ler_doubles, values \n");
    l = fgets(linha, SIZEOFLINHA, f);
    dados->sMat.values = ler_doubles(linha, &count);
    if (count != dados->sMat.nnz) {
        fprintf(stderr, "Erro: values com tamanho %d, esperado %d\n", count, dados->sMat.nnz);
        fclose(f); return NULL; }

    // 5ª linha - vInicial
    printf("ler_doubles, vInicial \n");
    l = fgets(linha, SIZEOFLINHA, f);
    dados->vInicial = ler_doubles(linha, &count);
    if (count != N) {
        fprintf(stderr, "Erro: vInicial com tamanho %d, esperado %d\n", count, N);
        fclose(f); return NULL; }
    //print_smatrix(&dados->sMat); exit(0);

    // 6ª linha - numIteracoes
    l = fgets(linha, SIZEOFLINHA, f);
    sscanf(linha, "%d", &dados->numIteracoes);
    printf("%d", dados->numIteracoes); 

    // Linhas seguintes - iterações
    dados->iteracoes = malloc(dados->numIteracoes * sizeof(double *));
    for (int i = 0; i < dados->numIteracoes; i++) {
        printf("ler_doubles, output das iteracoes \n");
        l = fgets(linha, SIZEOFLINHA, f);
        dados->iteracoes[i] = ler_doubles(linha, &count);
        if (count != N) {
            fprintf(stderr, "Erro: iteração %d tem %d valores, esperado %d\n", i, count, N);
            fclose(f);
            return NULL;
        }
    }
    fclose(f);
    free(linha);
    printf(" final ... em DadosCoined* lerDadosCoinedR(const char *fileName) {\n");
    return dados;
}

DadosCoined* lerDadosCoinedC(const char *fileName) {
    printf(" em DadosCoined*  lerDadosCoinedR(const char *fileName) {\n");

    printf(" arquivo de entrada: %s\n", fileName);
    FILE *f = fopen(fileName, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir o arquivo: %s\n", fileName);
        return NULL;
    }

    DadosCoined *dados = malloc(sizeof(DadosCoined));
    dados->sMat.type = T_COMPLEX;

    if (!dados) { fclose(f); return NULL; }

    int SIZEOFLINHA = 2048 * 2048 * 100;
    char *linha = (char*) malloc(SIZEOFLINHA);
    if (!linha) { fclose(f); free(dados); return NULL; }

    // 1ª linha - perm
    printf("ler perm\n");
    if (!fgets(linha, SIZEOFLINHA, f)) goto erro;
    dados->perm = ler_inteiros(linha, &dados->N);
    int N = dados->N;

    // 2ª linha - row_ptr
    printf("ler row_ptr\n");
    int n_row_ptr;
    if (!fgets(linha, SIZEOFLINHA, f)) goto erro;
    dados->sMat.row_ptr = ler_inteiros(linha, &n_row_ptr);
    dados->sMat.nrow = n_row_ptr - 1;
    dados->sMat.ncol = n_row_ptr - 1;
    dados->sMat.nnz  = dados->sMat.row_ptr[n_row_ptr - 1];

    // 3ª linha - col_idx
    printf("ler col_idx\n");
    int count;
    if (!fgets(linha, SIZEOFLINHA, f)) goto erro;
    dados->sMat.col_idx = ler_inteiros(linha, &count);
    if (count != dados->sMat.nnz) {
        fprintf(stderr, "Erro: col_idx com tamanho %d, esperado %d\n", count, dados->sMat.nnz);
        goto erro;
    }

    // 4ª linha - values (complexos)
    printf("ler complexos, values\n");
    if (!fgets(linha, SIZEOFLINHA, f)) goto erro;
    int n_complexos;
    double *valores = ler_complexos_como_doubles(linha, &n_complexos);
    if (n_complexos != dados->sMat.nnz) {
        fprintf(stderr, "Erro: values complexos com %d, esperado %d\n", n_complexos, dados->sMat.nnz);
        free(valores);
        goto erro;
    }
    dados->sMat.values = valores;

    // 5ª linha - vInicial (complexos)
    printf("ler complexos, vInicial\n");
    if (!fgets(linha, SIZEOFLINHA, f)) goto erro;
    dados->vInicial = ler_complexos_como_doubles(linha, &count);
    if (count != N) {
        fprintf(stderr, "Erro: vInicial com tamanho %d, esperado %d\n", count, N);
        goto erro;
    }

    // 6ª linha - numIteracoes
    if (!fgets(linha, SIZEOFLINHA, f)) goto erro;
    sscanf(linha, "%d", &dados->numIteracoes);

    printf(">> %d\n", dados->numIteracoes);

    // Linhas seguintes - iterações (complexas)
    dados->iteracoes = malloc(dados->numIteracoes * sizeof(double *));
    for (int i = 0; i < dados->numIteracoes; i++) {
        printf("ler complexos, output das iteracoes\n");
        if (!fgets(linha, SIZEOFLINHA, f)) goto erro;
        //printf("linha = %s\n", linha);
        dados->iteracoes[i] = ler_complexos_como_doubles(linha, &count);
        //printf(">> %d, %d\n", count, N);
        if (count != N) {
            fprintf(stderr, "Erro: iteração %d tem %d valores, esperado %d\n", i, count, N);
            goto erro;
        }
    }

    fclose(f);
    free(linha);
    printf(" final ... em DadosCoined* lerDadosCoinedC(const char *fileName) {\n");
    return dados;

erro:
    fprintf(stderr, "Erro durante a leitura do arquivo.\n");
    fclose(f);
    free(linha);
    free(dados);
    return NULL;
}

char arqEntradaA[]="entradasCoined_04_25/177R.txt";
char arqEntradaB[]="entradasCoined_04_25/129R.txt";
char arqEntradaC[]="entradasCoined_04_25/165R.txt";

static DadosCoined *dadosB;

int mainB(int argc, char *argv[]) {

    printf( " +++ em lerDados::main ( \n");
    if (argc != 4) {
        fprintf(stderr, "Uso: %s <arquivo.txt> <REPETICOES> <REPETICOESG>\n", argv[0]);
        return 1;
    }

    const char *arqEntrada = argv[1];
    int REPETICOES = atoi(argv[2]);

    if (REPETICOES <= 0) {
        fprintf(stderr, "Número de iterações inválido: %s\n", argv[2]);
        return 1;
    }

    dadosB = lerDadosCoinedR(arqEntrada);
    if (!dadosB) {
        fprintf(stderr, "Erro ao ler o arquivo: %s\n", arqEntrada);
        return 1;
    }

    print_smatrix(&dadosB->sMat);
    printf("N = %d, nnz = %d, Iterações no arquivo = %d, REPETICOES = %d\n",
           dadosB->N, dadosB->sMat.nnz, dadosB->numIteracoes, REPETICOES);

    return 0;
    for (int i = 0; i <  dadosB->numIteracoes; i++) {
        printf("Iteração %d:\n", i);
        for (int j = 0; j < dadosB->N; j++) {
            printf("%g ", dadosB->iteracoes[i][j]);
        }
        printf("\n");
    }

    // Liberação de memória (se necessário)
    // free_dadosCoined(dados);

}
