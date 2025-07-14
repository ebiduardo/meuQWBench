import sys
import re

# Define a enum data_type (assumido como int)
DATA_TYPE_COMPLEX = 3 #complex
DATA_TYPE_DOUBLE  = 2 #double

class SMatrix:
    def __init__(self):
        self.nrow = 0
        self.ncol = 0
        self.nnz = 0
        self.row_ptr = []
        self.col_idx = []
        self.values = []
        self.type = 0 

class DadosCoined:
    def __init__(self):
        self.perm = []
        self.N = 0
        self.sMat = SMatrix()
        self.vInicial = []
        self.numIteracoes = 0
        self.iteracoes = []

def ler_inteiros(line):
    return list(map(int, line.strip().split()))

def ler_doubles(line):
    return list(map(float, line.strip().split()))

def is_complex(line):
    return any(c in line for c in 'jJ()')

def ler_complexos(line):
    complex_numbers = []
    pattern = re.compile(
        r'(?:\(?\s*([+-]?\d*\.?\d+([eE][+-]?\d+)?)\s*([+-])\s*([+-]?\d*\.?\d+([eE][+-]?\d+)?)[jJ]\s*\)?)|'  # (a±bj)
        r'([+-]?\d*\.?\d+([eE][+-]?\d+)?)[jJ]|'                                                             # bj
        r'([+-]?\d*\.?\d+([eE][+-]?\d+)?)'                                                                   # a
    )
    
    for match in pattern.finditer(line):
        groups = match.groups()
        if groups[0]:  # Caso a±bj
            real = float(groups[0])
            imag = float(groups[3]) * (-1 if groups[2] == '-' else 1)
        elif groups[5]:  # Caso bj
            real = 0.0
            imag = float(groups[5])
        else:  # Caso a
            real = float(groups[7])
            imag = 0.0
        complex_numbers.extend([real, imag])
    
    return complex_numbers

def ler_linha_numeros(line):
    if is_complex(line):
        return ler_complexos(line), DATA_TYPE_COMPLEX
    else:
        return ler_doubles(line), DATA_TYPE_DOUBLE

def ler_dados_coined(file_name):
    dados = DadosCoined()
    
    with open(file_name, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Linha 1: perm
    dados.perm = ler_inteiros(lines[0])
    dados.N = len(dados.perm)
    
    # Linha 2: row_ptr
    row_ptr = ler_inteiros(lines[1])
    dados.sMat.row_ptr = row_ptr
    dados.sMat.nrow = len(row_ptr) - 1
    dados.sMat.ncol = dados.sMat.nrow
    dados.sMat.nnz = row_ptr[-1]
    
    # Linha 3: col_idx
    dados.sMat.col_idx = ler_inteiros(lines[2])
    if len(dados.sMat.col_idx) != dados.sMat.nnz:
        raise ValueError(f"col_idx size mismatch: {len(dados.sMat.col_idx)} != {dados.sMat.nnz}")
    
    # Linha 4: values
    values, data_type = ler_linha_numeros(lines[3])
    dados.sMat.type = data_type
    expected = 2 * dados.sMat.nnz if data_type == DATA_TYPE_COMPLEX else dados.sMat.nnz
    if len(values) != expected:
        raise ValueError(f"Values size mismatch: {len(values)} != {expected}")
    dados.sMat.values = values
    
    # Linha 5: vInicial
    vInicial, tipo = ler_linha_numeros(lines[4])
    if tipo != data_type:
        raise ValueError("Data type mismatch in vInicial")
    expected = 2 * dados.N if data_type == DATA_TYPE_COMPLEX else dados.N
    if len(vInicial) != expected:
        raise ValueError(f"vInicial size mismatch: {len(vInicial)} != {expected}")
    dados.vInicial = vInicial
    
    # Linha 6: numIteracoes
    dados.numIteracoes = int(lines[5].split()[0])
    
    # Iteracoes
    dados.iteracoes = []
    for i in range(6, 6 + dados.numIteracoes):
        valores, tipo = ler_linha_numeros(lines[i])
        if tipo != data_type:
            raise ValueError(f"Data type mismatch in iteration {i-6}")
        expected = 2 * dados.N if data_type == DATA_TYPE_COMPLEX else dados.N
        if len(valores) != expected:
            raise ValueError(f"Iteration {i-6} size mismatch: {len(valores)} != {expected}")
        dados.iteracoes.append(valores)
    
    return dados

def print_smatrix(smat):
    print(f"Matriz Esparsa ({smat.nrow}x{smat.ncol}, nnz={smat.nnz}, tipo={smat.type})")
    print(f"row_ptr: {smat.row_ptr[:5]}... (total {len(smat.row_ptr)})")
    print(f"col_idx: {smat.col_idx[:5]}... (total {len(smat.col_idx)})")
    print(f"values: {smat.values[:5]}... (total {len(smat.values)})")

def main():
    if len(sys.argv) != 3:
        print("Uso: python script.py <arquivo.txt> <REPETICOES>")
        sys.exit(1)
    
    try:
        dados = ler_dados_coined(sys.argv[1])
    except Exception as e:
        print(f"Erro: {str(e)}")
        sys.exit(1)
    
    print_smatrix(dados.sMat)
    print(f"\nN = {dados.N}, Iterações = {dados.numIteracoes}")
    print(f"Tipo de dados: {dados.sMat.type}")
    print(f"Primeira iteração: {dados.iteracoes[0][:4]}...")

#if __name__ == "__main__":
#    main()
