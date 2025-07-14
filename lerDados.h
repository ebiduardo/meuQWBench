typedef struct {
    int *perm;
    int N;
    smatrix_t sMat;
    double *vInicial;
    int numIteracoes;
    double **iteracoes;
} DadosCoined;

int *         ler_inteiros    (char *linha, int *out_count) ;
double *      ler_doubles     (char *linha, int *out_count) ;
DadosCoined * lerDadosCoinedR (const char *fileName) ;
DadosCoined * lerDadosCoinedC (const char *fileName) ;




