from scipy.sparse._sparsetools import csr_matvec
from scipy.sparse              import csr_matrix
import ctypes
import numpy as np
import scipy.sparse as sp 
import sys
import time

# carrega libgomp (em sistemas com GCC)
libomp = ctypes.CDLL("libiomp5.so")
#libomp = ctypes.CDLL("libgomp.so.1")
#libomp.omp_get_max_threads.restype = ctypes.c_int
#libomp.omp_get_num_threads.restype = ctypes.c_int

import lerDados as ler

#export LD_PRELOAD=/opt/intel/oneapi/2025.0/compiler/2025.0/lib/libiomp5.so
import os
#os.environ["LD_PRELOAD"] = "/opt/intel/oneapi/compiler/2025.0/lib/libiomp5.so"

# Carrega a biblioteca compartilhada
import platform
if platform.system() == "Windows":
    pass
    #libC = ctypes.CDLL(".\\libC_Algebra.dll")  # para Windows
else:
    pass
    libC = ctypes.CDLL("./libC_Algebra.so")   # para Linux/Mac

def verifyLibOMP():
    import os
    import subprocess
# Descobre o PID do processo atual
    pid = os.getpid()

# Procura por bibliotecas OpenMP carregadas (libgomp, libiomp)
    print(f"PID atual: {pid}")
    print("Bibliotecas OpenMP carregadas:")

    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            if "libgomp" in line or "libiomp" in line:
                print(line.strip())

verifyLibOMP(); #quit()


class SmatrixT(ctypes.Structure):
    _fields_ = [
        ("nrow", ctypes.c_int),
        ("ncol", ctypes.c_int),
        ("nnz", ctypes.c_int),
        ("row_ptr", ctypes.POINTER(ctypes.c_int)),
        ("col_idx", ctypes.POINTER(ctypes.c_int)),
        ("values", ctypes.POINTER(ctypes.c_double)),
        ("isPacked", ctypes.c_int),
        ("type", ctypes.c_int),          # Equivalente a data_type (int-enum)
        ("location", ctypes.c_ubyte),    # Corrigido: unsigned char -> c_ubyte
        ("extra", ctypes.c_void_p),
        ("idxColMem", ctypes.c_void_p)   # Corrigido: void* -> c_void_p
    ]

# União para data_vector_u
class DataVectorU(ctypes.Union):
    _fields_ = [
        ("v", ctypes.c_void_p),          # void*
        ("i", ctypes.POINTER(ctypes.c_int)),
        ("f", ctypes.POINTER(ctypes.c_double)),
        ("s", ctypes.POINTER(ctypes.c_void_p)),
    ]

# Estrutura vector_t corrigida
class VectorT(ctypes.Structure):
    _fields_ = [
        ("value", DataVectorU),
        ("len", ctypes.c_int),
        ("type", ctypes.c_int),          # data_type
        ("location", ctypes.c_ubyte),    # Corrigido: unsigned char -> c_ubyte
        ("extra", ctypes.c_void_p),
        ("externalData", ctypes.c_int)
    ]

# Protótipos das funções

libC.smatrix_vector_mult.argtypes = [ctypes.POINTER(SmatrixT), ctypes.POINTER(VectorT), ctypes.POINTER(VectorT)]
libC.smatrix_vector_mult.restype  = None

libC.permute_and_multiply.argtypes = [ctypes.POINTER(SmatrixT), ctypes.POINTER(SmatrixT), ctypes.POINTER(SmatrixT)]
libC.permute_and_multiply.restype  = None

libC.compute_U_row_ptr.argtypes = [ctypes.POINTER(SmatrixT), ctypes.POINTER(SmatrixT), ctypes.POINTER(SmatrixT)]
libC.compute_U_row_ptr.restype  = None

libC.print_smatrix.argtypes = [ctypes.POINTER(SmatrixT)]
libC.print_smatrix.restype  = None 

libC.print_vector.argtypes = [ctypes.POINTER(VectorT), ctypes.c_int, ctypes.c_int]
libC.print_vector.restype = None  # já que a função C é void

libC.mostrarConteudoV.argtypes = [ctypes.POINTER(VectorT), ctypes.POINTER(ctypes.c_char)]
libC.mostrarConteudoV.restype = None  # já que a função C é void

libC.print_vector.argtypes = [ctypes.POINTER(VectorT)]
libC.print_vector.restype = None  # já que a função C é void


def atribuir_dados():
    print("\n em atribuir_dados")
    # Permutação
    perm = np.array([5, 2, 1, 8, 9, 0, 7, 6, 3, 4])
    # Dados da matriz CSR
    indptr  = np.array([0, 1, 2, 5, 8, 11, 12, 13, 14, 15, 16])
    indices = np.array([0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9])
    data    = np.array([-1.0, -1.0, -1.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ])
    vI      = np.array([0.31622776601683794] * 10)
    numIteracoes=80
    n_iter=2
    r1 = np.array([-0.31622776601683794] * 10)
    r2 = np.array([ 0.31622776601683794] * 10)
    return perm, indptr, indices, data, vI, n_iter, r2


def carregar_dados(nome_arquivo):
    with open(nome_arquivo, 'r') as f:
        linhas = [linha.strip() for linha in f if linha.strip()]
    print("\n em carregar_dados:", nome_arquivo)

    perm     = np.fromstring(linhas[0], sep=' ', dtype=int)
    indptr   = np.fromstring(linhas[1], sep=' ', dtype=int)
    indices  = np.fromstring(linhas[2], sep=' ', dtype=int)
    #data     = np.fromstring(linhas[3], sep=' ', dtype=float)

    valores_str = linhas[3].strip().split()
    data     = np.array(valores_str, dtype=float)

    vI_Floats   = np.fromstring(linhas[4], sep=' ', dtype=float)

    n_iter   = 1
    n_iter   = np.fromstring(linhas[5], sep=' ', dtype=int) 
    numIteracoes=n_iter[0]
    i=1;
    print(" lendo vetores de saida, .....", i)
    r_ultIter_Floats=np.fromstring(linhas[5+i], sep=' ', dtype=float)
    for i in range(2,(numIteracoes+1)):
        print(" lendo vetores de saida, .....", i)
        r_ultIter_Floats=np.fromstring(linhas[5+i], sep=' ', dtype=float)
        vT_c   = r_ultIter_Floats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vT = VectorT( value = DataVectorU(f=vT_c), len = len(r_ultIter_Floats), type = DATA_TYPE_DOUBLE,
           location = 0, extra = None, externalData = 1 )
        #libC.mostrarConteudoV(ctypes.byref(vT), ctypes.c_char_p(b" Vetor da ultima iteracao:"))
 
    return perm, indptr, indices, data, vI_Floats, numIteracoes, r_ultIter_Floats

def mostrar_vetor(vetor, label_, max_elementos=5):
    np.set_printoptions(precision=4, suppress=True, linewidth=180, sign=' ')  # ' ' força espaço antes dos positivos
    tamanho = len(vetor)
    print(f"  {label_} (tamanho...={tamanho}):")
    if tamanho <= 2 * max_elementos:
        print(vetor)
    else:
        inicio = vetor[:max_elementos]
        fim    = vetor[-max_elementos:]
        print(f"[{', '.join(f'{v:.6g}' for v in inicio)} ... {', '.join(f'{v:.6g}' for v in fim)}]")
    return

import ctypes

def mostrar_dados(entrada, perm, indptr, indices, data, vI_, n_iter, r_ultIter, tipo):
    print("Origem dos dados:", entrada)
    print("N        = ", len(perm))  # Corrigido para usar len() em lista
    print("nonZeros = ", len(indices))  # Corrigido para usar len() em lista

    mostrar_vetor(perm, " vetor permutacao, ")

    # Convertendo lista Python para array C de doubles
    vI_array = (ctypes.c_double * len(vI_))(*vI_)
    vI_c = ctypes.cast(vI_array, ctypes.POINTER(ctypes.c_double))
    
    vI = VectorT(
        value=DataVectorU(f=vI_c),
        len=len(perm), # Manter o tamanho para complexos tbm
        type=tipo,
        location=0,
        extra=None,
        externalData=1
    )
    libC.mostrarConteudoV(ctypes.byref(vI), ctypes.c_char_p(b"Vetor inicial           :"))

    print("\nNúmero de iterações (n_iter):", n_iter)

    # Convertendo lista Python para array C de doubles
    vT_array = (ctypes.c_double * len(r_ultIter))(*r_ultIter)
    vT_c = ctypes.cast(vT_array, ctypes.POINTER(ctypes.c_double))
    
    vT = VectorT( value=DataVectorU(f=vT_c), len=len(perm), type=tipo,
        location=0, extra=None, externalData=1)
    libC.mostrarConteudoV(ctypes.byref(vT), ctypes.c_char_p(b"Vetor output da ultima iteracao:"))

    return

def setDados():
    #dados = []
    libC.disable_stdout_buffering()

    dirEntrada="entradasCoined_05_25"; fileName="R-177.txt"
    entrada=dirEntrada+"/"+fileName
    REPETICOES=5
    libOp="comSciPy"; comSciPy=False;
    comMKL=False;

    if len(sys.argv) < 4:
        print("Uso: python ", sys.argv[0], " <arquivo_de_entrada> <REPETICOES> <comSciPy> ")
        print("Uso: python ", sys.argv[0], entrada, REPETICOES,libOp)
        print("Uso: python ", sys.argv[0], " default")
        sys.exit(1)

    if sys.argv[1]=="default" : 
        pass
        #perm, indptr, indices, values, vI_f, numIteracoes, r_ultIter = atribuir_dados()
    else:
        print("carregar_dados(")
        entrada = sys.argv[1]
        print(" arquivo de entrada: ", entrada)
        #perm, indptr, indices, values, vI_Floats, numIteracoes, r_ultIter_Floats = carregar_dados(entrada)
        dados = ler.ler_dados_coined(entrada)

    if len(sys.argv) >= 3 : REPETICOES=int(sys.argv[2])

    libCNormal = False
    labelOp=   "algebra via  libC: permute_and_multiply; smatrix_vector_mult"
    comSciPy=False;
    if len(sys.argv) >= 4 : 
     if sys.argv[3] == "comSciPy":
       comSciPy=True;
       labelOp="algebra via SciPy: U=S_.dot(C_); y = U.dot(x)"
     if sys.argv[3] == "comMKL":
       comMKL=True;
       labelOp="algebra via   MKL: U=dot_product_mkl(S_, C_); y=dot_product_mkl(U, x)"

    #print(".......", len(sys.argv), sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3])
    #print(libOp,  " = ", comSciPy, ", ", labelOp);


    mostrar_dados(entrada, dados.perm, dados.sMat.row_ptr, dados.sMat.col_idx, dados.sMat.values, dados.vInicial, dados.numIteracoes, dados.iteracoes[-1],  dados.sMat.type)


    #print(">>: ", dados.vInicial)

    # Conversão do vetor inicial
    vI_array = (ctypes.c_double * len(dados.vInicial))(*dados.vInicial)
    vI_c = ctypes.cast(vI_array, ctypes.POINTER(ctypes.c_double))
    
    vIt = VectorT( value=DataVectorU(f=vI_c), len=dados.N,  # Usando dados.N ao invés de len(perm)
        type=dados.sMat.type, location=0, extra=None, externalData=1)

    # Conversão do vetor de última iteração
    ultima_iter = dados.iteracoes[-1]
    vO_array = (ctypes.c_double * len(ultima_iter))(*ultima_iter)
    vO_c = ctypes.cast(vO_array, ctypes.POINTER(ctypes.c_double))
    
    vOt = VectorT( value=DataVectorU(f=vO_c), len=dados.N,  # Usando dados.N type=dados.sMat.type,
        location=0, extra=None, externalData=1)
    
    
    vI_c = np.array(dados.vInicial).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    vIt = VectorT( value = DataVectorU(f=vI_c), len = len(dados.perm), type = dados.sMat.type,
           location = 0, extra = None, externalData = 1 )
    libC.mostrarConteudoV(ctypes.byref(vIt), ctypes.c_char_p(b" Vetor inicial         Lido:"))

    vO_c = np.array(dados.iteracoes[-1]).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    vOt = VectorT( value = DataVectorU(f=vO_c), len = len(dados.perm), type = dados.sMat.type,
           location = 0, extra = None, externalData = 1 )
    libC.mostrarConteudoV(ctypes.byref(vOt), ctypes.c_char_p(b" Vetor output final    Lido:"))

    N = len(dados.perm)
    row_ptr = np.arange(N + 1, dtype=np.int32)   # Ponteiros para início de cada linha
    col_idx = np.array(dados.perm, dtype=np.int32)  # Permutação como índices de coluna

    if dados.sMat.type == ler.DATA_TYPE_DOUBLE:
        valuesI = np.ones(N, dtype=np.float64)
    else:
        valuesI = np.ones(N, dtype=np.complex128)

    S = csr_matrix((valuesI, col_idx, row_ptr), shape=(N, N))

    smatS = SmatrixT(
        nrow      = S.shape[0], ncol      = S.shape[1], nnz       = S.nnz,
        row_ptr   = S.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        col_idx   = S.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values    = S.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), isPacked  = 1, type      = dados.sMat.type,
        location  = 0, extra     = None, idxColMem = None)
    if( N  < 25 ): print("\n Matriz S         :\n"); libC.print_smatrix(smatS)

    # Verificar se é complexo ANTES de criar a matriz
    is_complex = (dados.sMat.type == ler.DATA_TYPE_COMPLEX)

    if dados.sMat.type == ler.DATA_TYPE_DOUBLE:
        # Matriz real - valores e índices devem ter mesmo tamanho
        assert len(dados.sMat.values) == len(dados.sMat.col_idx), \
            f"Tamanhos inconsistentes: values={len(dados.sMat.values)}, col_idx={len(dados.sMat.col_idx)}"
        values = np.ctypeslib.as_array(dados.sMat.values).view(np.float64)
    else:
        print("Matriz complexa detectada - convertendo formato intercalado")
        nnz = len(dados.sMat.col_idx)  # Número real de elementos não-zero
        assert len(dados.sMat.values) == 2 * nnz, \
            f"Para complexo, esperado 2*{nnz}={2*nnz} valores, temos {len(dados.sMat.values)}"
        values = np.ctypeslib.as_array(dados.sMat.values).view(np.complex128)

    row_ptr = np.ctypeslib.as_array(dados.sMat.row_ptr, shape=(dados.N + 1,))
    col_idx = np.ctypeslib.as_array(dados.sMat.col_idx, shape=(len(values),))
    C = csr_matrix((values, col_idx, row_ptr), shape=(dados.N, dados.N))

    if N < 25: print("\n Matriz C_:"); print(C.toarray())

    # Cria um SmatrixT apontando para os dados da matriz C
    smatC = SmatrixT(
        nrow      = C.shape[0], ncol      = C.shape[1], nnz       = C.nnz,
        row_ptr   = C.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        col_idx   = C.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values    = C.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        isPacked  = 1, type      = dados.sMat.type,
        location  = 0, extra     = None, idxColMem = None
    )

    if( N  < 25 ): print("\n Matriz C         :\n"); libC.print_smatrix(smatC)

    print("em setDados, comSciPy=",comSciPy);

    return smatS, S, smatC, C, dados.vInicial, dados.numIteracoes, REPETICOES, comSciPy, comMKL, labelOp, libCNormal

def simulateSciPy(S_, C_, vI_Floats_, numIteracoes_, REPETICOES_):
    """
    Executa a simulação do modelo SciPy 
    S_, C_ são matrizes scipy.sparse.csr_matrix.
    vI_Floats_ é o vetor inicial (numpy array).
    perm_ é a permutação das linhas (numpy array de inteiros).
    """
    tempo_medidoS = 0.0; tempo_medidoI = 0.0

    tipo = ler.DATA_TYPE_DOUBLE
    if S_.dtype == np.complex128:
       tipo = ler.DATA_TYPE_COMPLEX

    copiaVetorInicial = vI_Floats_.copy()

    for r in range(1, REPETICOES_ + 1):
        # Verificação para imprimir apenas na primeira e última repetição
        if r == 1 or r == REPETICOES_:
            print(f"++++++++++++++++ REPETIÇÃO {r} de {REPETICOES_}")  # , flush=True
        
        # Vetor de entrada  - reinicializado a cada repetição
        vIf = np.array(copiaVetorInicial, dtype=np.float64)
        v_input = VectorT( value=DataVectorU(f=vIf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))), len=S_.shape[0], type=tipo,
            location=0, extra=None, externalData=1)
        #libC.mostrarConteudoV( ctypes.byref(v_input),   ctypes.c_char_p(b"em simulate,A     v_input"))

        # Vetor de saida  - reinicializado a cada repetição
        vB = np.zeros_like(copiaVetorInicial)
        v_output = VectorT( value=DataVectorU(f=vB.ctypes.data_as(ctypes.POINTER(ctypes.c_double))), len=v_input.len, type=tipo,
            location=0, extra=None, externalData=1)
        #libC.mostrarConteudoV( ctypes.byref(v_output),  ctypes.c_char_p(b"em simulate,A    v_output"))

        # Etapa de permutação ou montagem de U
        tempo_inicialS = time.perf_counter()

        U = S_.dot(C_)
        tempo_finalS   = time.perf_counter()
        tempo_medidoS += (tempo_finalS - tempo_inicialS)

        # Iterações de multiplicação vetor = matriz * vetor

        if tipo == ler.DATA_TYPE_DOUBLE:
            v_input_np  = np.ctypeslib.as_array(vIf, shape=(v_input.len,))
            v_output_np = np.ctypeslib.as_array( vB, shape=(v_input.len,))
        else:
            v_input_np  = np.ctypeslib.as_array(vIf.view(np.complex128), shape=(v_input.len,))
            v_output_np = np.ctypeslib.as_array( vB.view(np.complex128), shape=(v_input.len,))

        tempo_inicialI = time.perf_counter()

        for i in range(1, numIteracoes_ + 1):

            v_output_np[:] = U.dot(v_input_np)

            if ( i==1 or i == numIteracoes_) and (r==1 or r== REPETICOES_) :
               label=f"{i}/{numIteracoes_}_{r}"+", em simulate,  v_input"
               libC.mostrarConteudoV( ctypes.byref(v_input),  ctypes.c_char_p(label.encode()))
               label=f"{i}/{numIteracoes_}_{r}"+", em simulate, v_output"
               libC.mostrarConteudoV( ctypes.byref(v_output), ctypes.c_char_p(label.encode()))
            
            # Troca entrada <-> saída
            v_input, v_output       = v_output, v_input
            v_input_np, v_output_np = v_output_np, v_input_np

        tempo_finalI = time.perf_counter()
        tempo_medidoI += (tempo_finalI - tempo_inicialI)

    # Relatórios
    print(f"Tempo total de {REPETICOES_} permutacoes: {tempo_medidoS:.6f} s")
    print(f"Tempo medio por permutacao: {tempo_medidoS/REPETICOES_:.6f} s")
    print(f"Tempo total de {REPETICOES_ * numIteracoes_} iteracoes: {tempo_medidoI:.6f} s")
    print(f"Tempo medio por iteracao  : {tempo_medidoI/(REPETICOES_*numIteracoes_):.6f} s")

#TBD (Por enquanto é só uma cópia do simulateSciPy)
def simulateMKL(S_, C_, vI_Floats_, numIteracoes_, REPETICOES_, num_threads=1):
    """
    Executa a simulação do modelo SciPy 
    S_, C_ são matrizes scipy.sparse.csr_matrix.
    vI_Floats_ é o vetor inicial (numpy array).
    perm_ é a permutação das linhas (numpy array de inteiros).
    """

    from sparse_dot_mkl import dot_product_mkl, mkl_set_num_threads, mkl_get_max_threads
    #mkl.mkl_set_num_threads(c_int(4))
    #num_threads=4
    #mkl_set_num_threads(num_threads)

    #print(f"Threads MKL ativos: {mkl_get_max_threads()}")
    print("MKL num max threads:", mkl_get_max_threads())
    
    tempo_medidoS = 0.0; tempo_medidoI = 0.0

    tipo = ler.DATA_TYPE_DOUBLE
    if S_.dtype == np.complex128:
       tipo = ler.DATA_TYPE_COMPLEX

    copiaVetorInicial = vI_Floats_.copy()

    for r in range(1, REPETICOES_ + 1):
        # Verificação para imprimir apenas na primeira e última repetição
        if r == 1 or r == REPETICOES_:
            print(f"++++++++++++++++ REPETIÇÃO {r} de {REPETICOES_}")  # , flush=True
        
        # Vetor de entrada  - reinicializado a cada repetição
        vIf = np.array(copiaVetorInicial, dtype=np.float64)
        v_input = VectorT( value=DataVectorU(f=vIf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))), len=S_.shape[0], type=tipo,
            location=0, extra=None, externalData=1)
        #libC.mostrarConteudoV( ctypes.byref(v_input),   ctypes.c_char_p(b"em simulate,A     v_input"))

        # Vetor de saida  - reinicializado a cada repetição
        vB = np.zeros_like(copiaVetorInicial)
        v_output = VectorT( value=DataVectorU(f=vB.ctypes.data_as(ctypes.POINTER(ctypes.c_double))), len=v_input.len, type=tipo,
            location=0, extra=None, externalData=1)
        #libC.mostrarConteudoV( ctypes.byref(v_output),  ctypes.c_char_p(b"em simulate,A    v_output"))

        # Etapa de permutação ou montagem de U
        tempo_inicialS = time.perf_counter()

        #U = S_.dot(C_)
        U = dot_product_mkl(S_, C_)
        tempo_finalS   = time.perf_counter()
        tempo_medidoS += (tempo_finalS - tempo_inicialS)

        # Iterações de multiplicação vetor = matriz * vetor

        if tipo == ler.DATA_TYPE_DOUBLE:
            v_input_np  = np.ctypeslib.as_array(vIf, shape=(v_input.len,))
            v_output_np = np.ctypeslib.as_array( vB, shape=(v_input.len,))
        else:
            v_input_np  = np.ctypeslib.as_array(vIf.view(np.complex128), shape=(v_input.len,))
            v_output_np = np.ctypeslib.as_array( vB.view(np.complex128), shape=(v_input.len,))

        tempo_inicialI = time.perf_counter()

        for i in range(1, numIteracoes_ + 1):

            #v_output_np[:] = U.dot(v_input_np)
            v_output_np[:] = dot_product_mkl(U, v_input_np)

            if ( i==1 or i == numIteracoes_) and (r==1 or r== REPETICOES_) :
               label=f"{i}/{numIteracoes_}_{r}"+", em simulate,  v_input"
               libC.mostrarConteudoV( ctypes.byref(v_input),  ctypes.c_char_p(label.encode()))
               label=f"{i}/{numIteracoes_}_{r}"+", em simulate, v_output"
               libC.mostrarConteudoV( ctypes.byref(v_output), ctypes.c_char_p(label.encode()))
            
            # Troca entrada <-> saída
            v_input, v_output       = v_output, v_input
            v_input_np, v_output_np = v_output_np, v_input_np

        tempo_finalI = time.perf_counter()
        tempo_medidoI += (tempo_finalI - tempo_inicialI)

    # Relatórios
    print(f"Tempo total de {REPETICOES_} permutacoes: {tempo_medidoS:.6f} s")
    print(f"Tempo medio por permutacao: {tempo_medidoS/REPETICOES_:.6f} s")
    print(f"Tempo total de {REPETICOES_ * numIteracoes_} iteracoes: {tempo_medidoI:.6f} s")
    print(f"Tempo medio por iteracao  : {tempo_medidoI/(REPETICOES_*numIteracoes_):.6f} s")

def copy_smatrix(smat_src_):

    smat_dest = SmatrixT()

    smat_dest.type = smat_src_.type
    smat_dest.nrow = smat_src_.nrow
    nrow           = smat_dest.nrow
    smat_dest.ncol = smat_src_.ncol
    smat_dest.nnz  = smat_src_.nnz

    nnz            = 1*smat_dest.nnz
    if smat_src_.type == ler.DATA_TYPE_COMPLEX: nnz=2*smat_dest.nnz


    # 🔸 Copiar values (double * nnz)
    values_array = (ctypes.c_double * nnz)()
    ctypes.memmove(values_array, smat_src_.values, ctypes.sizeof(ctypes.c_double) * nnz)
    smat_dest.values = ctypes.cast(values_array, ctypes.POINTER(ctypes.c_double))

    # 🔸 Copiar col_idx (int * nnz)
    col_idx_array = (ctypes.c_int * nnz)()
    ctypes.memmove(col_idx_array, smat_src_.col_idx, ctypes.sizeof(ctypes.c_int) * nnz)
    smat_dest.col_idx = ctypes.cast(col_idx_array, ctypes.POINTER(ctypes.c_int))

    # 🔸 Copiar row_ptr (int * (nrow + 1))
    row_ptr_array = (ctypes.c_int * (nrow + 1))()
    ctypes.memmove(row_ptr_array, smat_src_.row_ptr, ctypes.sizeof(ctypes.c_int) * (nrow + 1))
    smat_dest.row_ptr = ctypes.cast(row_ptr_array, ctypes.POINTER(ctypes.c_int))

    return smat_dest


def simulateLibC(smatS_, smatC_, vI_Floats_,  numIteracoes_, REPETICOES_, libCNormal):
    """
    Executa a simulação do modelo comutando entre backends  C.
    smatS_, smatC_ são matrizes no formato da libC.
    vI_Floats_ é o vetor inicial (numpy array).
    perm_ é a permutação das linhas (numpy array de inteiros).
    """

    print("OMP num max threads:", libomp.omp_get_max_threads())
    # Preparar vetor inicial (compartilhado entre repetições)
    copiaVetorInicial = vI_Floats_.copy()
    # Preparar matriz U no backend desejado (sem cópias)
    smatU = copy_smatrix(smatC_)
    #print("em simulateLibC, smatU_:\n"); libC.print_smatrix( ctypes.byref(smatU))

    tempo_medidoS = 0.0; tempo_medidoI = 0.0
    for r in range(1, REPETICOES_ + 1):
#       smatU = copy_smatrix(smatUCopia)
        smatU = copy_smatrix(smatC_)
        # Verificação para imprimir apenas na primeira e última repetição
        if r == 1 or r == REPETICOES_:
            print(f"++++++++++++++++ REPETIÇÃO {r} de {REPETICOES_}")  # , flush=True
        

        # Vetor de entrada  - reinicializado a cada repetição
        vIf     = np.array(copiaVetorInicial.copy(), dtype=np.float64)
        v_input = VectorT( value=DataVectorU(f=vIf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))), len=smatS_.nrow, type=smatC_.type,
            location=0, extra=None, externalData=1)

        # Vetor de saida  - reinicializado a cada repetição
        vB       = np.zeros(len(copiaVetorInicial), dtype=np.float64)  # Adicionado dtype explícito
        v_output = VectorT( value=DataVectorU(f=vB.ctypes.data_as(ctypes.POINTER(ctypes.c_double))), len=v_input.len, type=v_input.type,
            location=0, extra=None, externalData=1)

        tempo_inicialS = time.perf_counter()
        # Etapa de permutação ou montagem de U
        libC.compute_U_row_ptr   (ctypes.byref(smatS_), ctypes.byref(smatC_), ctypes.byref(smatU))
        
        libC.permute_and_multiply(ctypes.byref(smatS_), ctypes.byref(smatC_), ctypes.byref(smatU))
        tempo_finalS   = time.perf_counter()
        tempo_medidoS += (tempo_finalS - tempo_inicialS)

        # Iterações de multiplicação vetor = matriz * vetor
        tempo_inicialI = time.perf_counter()
        
        for i in range(1, numIteracoes_ + 1):
            libC.smatrix_vector_mult( ctypes.byref(smatU), ctypes.byref(v_input), ctypes.byref(v_output))

            if ( i==1 or i == numIteracoes_) and (r==1 or r== REPETICOES_) :
               label=f"{i}/{numIteracoes_}_{r}"+", em simulate,  v_input"
               libC.mostrarConteudoV( ctypes.byref(v_input),  ctypes.c_char_p(label.encode()))
               label=f"{i}/{numIteracoes_}_{r}"+", em simulate, v_output"
               libC.mostrarConteudoV( ctypes.byref(v_output), ctypes.c_char_p(label.encode()))
            # Troca entrada <-> saída
            v_input, v_output       = v_output, v_input

        tempo_finalI = time.perf_counter()
        tempo_medidoI += (tempo_finalI - tempo_inicialI)

    # Relatórios
    print(f"Tempo total de {REPETICOES_} permutacoes: {tempo_medidoS:.6f} s")
    print(f"Tempo medio por permutacao: {tempo_medidoS/REPETICOES_:.6f} s")
    print(f"Tempo total de {REPETICOES_ * numIteracoes_} iteracoes: {tempo_medidoI:.6f} s")
    print(f"Tempo medio por iteracao  : {tempo_medidoI/(REPETICOES_*numIteracoes_):.6f} s")

def main():
   sys.stdout.flush()
#   stdout = ctypes.c_void_p.in_dll(libC, "stdout")
#   libC.fflush(stdout)

#   libc = ctypes.CDLL(None)
#   stdout = ctypes.c_void_p.in_dll(libc, "stdout")
#   libc.fflush(stdout)

   smatS,S,smatC,C,vI_Floats,numIteracoes,REPETICOES, comSciPy, comMKL, labelOp, libCNormal = setDados()

   nMt=libomp.omp_get_max_threads()
   print("OMP num max threads:", nMt)
   print(f"numMaxThrs= {nMt}, N= {smatC.nrow}, nnz= {smatC.nnz}, densidade= {100*(smatC.nnz/(smatC.nrow*smatC.nrow)):.4f} %",end="")

   print(f",  py, {labelOp}")

   if comSciPy:
      simulateSciPy(    S,     C, vI_Floats, numIteracoes, REPETICOES)
   elif comMKL:
      simulateMKL  (    S,     C, vI_Floats, numIteracoes, REPETICOES) # , 1)
   else:
      simulateLibC (smatS, smatC, vI_Floats, numIteracoes, REPETICOES, libCNormal)

main()
