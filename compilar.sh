MKLROOT=/usr/lib/x86_64-linux-gnu
MKLROOT=${ONEAPI_ROOT}/mkl/2025.0

#${ONEAPI_ROOT}/2025.0/include

SUF=I; CC=icx; CFLAGS="-c -fPIC -std=c11 -O3 -ffast-math -qopenmp "; LFLAGS="-qopenmp"
#LIBSI="   -L${MKLROOT}/lib/intel64 \
LIBSI="   -L${MKLROOT} \
    -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread  -liomp5  -lpthread -lm -ldl"
LIBS=$LIBSI

#INC="-I/usr/include/mkl"
INC="$MKLROOT/include"

SUF=G; CC=gcc; CFLAGS="-c -I$INC -fPIC -std=c11     -Ofast -fopenmp"; LFLAGS="-fopenmp"
LIBS="$LIBSG $LIBSI"


fonteC=coined
rm $fonteC$SUF.exe  ./libC_Algebra$SUF.so
# Compilar os arquivos .c em objetos
$CC $CFLAGS lerDados.c
$CC $CFLAGS $fonteC.c 
#$CC $LFLAGS lerDados.o $fonteC.o -o libC_Algebra$SUF.so

#criar lib SO
$CC -fPIC -shared -o libC_Algebra$SUF.so  lerDados.o $fonteC.o\
    -I${MKLROOT}/include $LIBS

#criar executavel
$CC $LFLAGS -o $fonteC$SUF.exe $fonteC.o ./libC_Algebra$SUF.so $LIBS

echo "resultado: libC_Algebra$SUF.so coined$SUF.exe"
#faz uma copia que está sendo usada no coined.py
cp libC_Algebra$SUF.so libC_Algebra.so
cp coined$SUF.exe coined.exe
echo "copiado para: libC_Algebra.so coined.exe"

#for d in $(echo $LD_LIBRARY_PATH |sed 's/:/\n/g') ; do echo $d; find $d -name libiomp5.so ;  done
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/2025.0/compiler/2025.0/lib

rm *.o
