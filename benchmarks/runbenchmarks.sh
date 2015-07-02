#!/bin/bash

BENCHMARK_DATE=$(date --iso-8601=minutes -u)
CPU_MODEL=$(cat /proc/cpuinfo | grep '^model name' | head -1 | sed 's/[^:]*: //')
BLAS_LIBRARY=$(ls /opt/OpenBLAS/lib/libopenblas*[0-9]*.so | head -1 | sed 's/.*\///')
N_ITERS=1000
N_SAMPLES=193  # twospirals-train


echo "Notch vs Torch7"
echo "==============="
echo

echo "Benchmark date:   $BENCHMARK_DATE" ; echo
echo "Benchmark CPU:    $CPU_MODEL" ; echo
echo "Benchmark OS:     $(uname -sr)" ; echo
echo "C++ compiler:     $(g++ --version | head -1)" ; echo
echo "BLAS:             ${BLAS_LIBRARY}" ; echo
echo

echo "Benchmarks"
echo "----------"
echo
echo " * twospirals: a 2-50-50-50-50-10-1 multilayer perceptron"
echo
echo

rm *.err *.log
RESULTS=./benchmark_results.txt
test -f $RESULTS && rm $RESULTS
>$RESULTS  printf "|%-30s|%10s|%10s|\n" "library and task" "time, s" "samples/s"
>>$RESULTS printf "|:-----------------------------|---------:|---------:|\n"

export LC_NUMERIC=C

(
    # requires torch7 (th) to be installed
    which th > /dev/null
    HAS_TORCH=$?

    if [[ $HAS_TORCH -eq 0 ]]; then
        echo -n "Torch SGD $N_ITERS iters... " ;
        time -p ( th bench_twospirals.lua $N_ITERS > torch.log ) 2> torch.err \
            && ( echo OK ; \
                 TORCH_TIME=$(awk '/^real /{print $2;}' torch.err) ; \
                 TORCH_SPS=$(awk "BEGIN {print $N_ITERS * $N_SAMPLES / $TORCH_TIME;}") ; \
                 >> $RESULTS printf "|%-30s|%10.2f|%10.0f|\n" \
                             "Torch7 $N_ITERS SGD iters" $TORCH_TIME $TORCH_SPS) \
            || ( echo "FAILED # see torch.err" )
    fi
) >&2

(
    # requires GNU C++ compiler and OpenBLAS in /opt/OpenBLAS
    test -f ./bench_twospirals && rm ./bench_twospirals
    g++ -std=c++11 -O3 -I.. -DNOTCH_USE_CBLAS bench_twospirals.cpp \
        -L/opt/OpenBLAS/lib -lopenblas -o bench_twospirals \
        || (echo "Notch build with BLAS FAILED")
    HAS_NOTCH=$(test -f ./bench_twospirals)

    if [[ $HAS_NOTCH -eq 0 ]]; then
        echo -n "Notch SGD $N_ITERS iters... " ;
        time -p ( ./bench_twospirals $N_ITERS > notch_sgd.log ) 2> notch_sgd.err \
            && ( echo OK ; \
                 EXE_TIME=$(awk '/^real /{print $2;}' notch_sgd.err) ; \
                 EXE_SPS=$(awk "BEGIN {print $N_ITERS * $N_SAMPLES / $EXE_TIME;}") ; \
                 >> $RESULTS printf "|%-30s|%10.2f|%10.0f|\n" \
                             "Notch $N_ITERS SGD iters" $EXE_TIME $EXE_SPS) \
            || ( echo "FAILED # see notch_sgd.err" )
    fi

    if [[ $HAS_NOTCH -eq 0 ]]; then
        echo -n "Notch ADADELTA $N_ITERS iters... " ;
        time -p ( ./bench_twospirals $N_ITERS --adadelta > notch_adadelta.log ) 2> notch_adadelta.err  \
            && ( echo OK ; \
                 EXE_TIME=$(awk '/^real /{print $2;}' notch_adadelta.err ) ; \
                 EXE_SPS=$(awk "BEGIN {print $N_ITERS * $N_SAMPLES / $EXE_TIME;}") ; \
                 >> $RESULTS printf "|%-30s|%10.2f|%10.0f|\n" \
                             "Notch $N_ITERS ADADELTA iters" $EXE_TIME $EXE_SPS) \
            || ( echo "FAILED # see notch_adadelta.err" )
    fi
) >&2

(
    # requires only GNU C++ compiler
    test -f ./bench_twospirals_noblas && rm ./bench_twospirals_noblas
    g++ -std=c++11 -O3 -I.. -fopenmp -DNOTCH_USE_OPENMP \
        bench_twospirals.cpp -o bench_twospirals_noblas \
        || (echo "Notch build without BLAS FAILED")
    HAS_NOTCH=$(test -f ./bench_twospirals_noblas)

    if [[ $HAS_NOTCH -eq 0 ]]; then
        echo -n "Notch (no BLAS) SGD $N_ITERS iters... " ;
        time -p ( ./bench_twospirals_noblas $N_ITERS > notch_noblas_sgd.log ) 2> notch_noblas_sgd.err \
            && ( echo OK ; \
                 EXE_TIME=$(awk '/^real /{print $2;}' notch_noblas_sgd.err) ; \
                 EXE_SPS=$(awk "BEGIN {print $N_ITERS * $N_SAMPLES / $EXE_TIME;}") ; \
                 >> $RESULTS printf "|%-30s|%10.2f|%10.0f|\n" \
                             "Notch (no BLAS) $N_ITERS SGD iters" $EXE_TIME $EXE_SPS) \
            || ( echo "FAILED # see notch_noblas_sgd.err" )
    fi

) >&2

echo
echo "Results"
echo "-------"
echo
cat $RESULTS
echo
