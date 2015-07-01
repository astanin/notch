#!/bin/bash

BENCHMARK_DATE=$(date --iso-8601=minutes -u)
CPU_MODEL=$(cat /proc/cpuinfo | grep '^model name' | head -1 | sed 's/[^:]*: //')
BLAS_LIBRARY=$(ls /opt/OpenBLAS/lib/libopenblas*.so | sort  | head -1 | sed 's/.*\///')


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

echo "Log"
echo "---"
echo
echo '```'
echo -n "Torch7 twospirals SGD 1000 iters: "

# requires torch7 (th) to be installed
time -p ( th bench_twospirals.lua > torch.log ) 2> torch.err \
    || ( echo -n "FAILED " ; echo "# see torch.err" ) \
    && ( echo -n "OK " ; tail -1 torch.log ;
         TORCH_TIME=$(awk '/^real /{print $2;}' torch.err)
         echo "Torch7 twospirals SGD real time: " $TORCH_TIME)

# requires GNU C++ compiler and OpenBLAS in /opt/OpenBLAS
test -f ./bench_twospirals && rm ./bench_twospirals
g++ -std=c++11 -O3 -I.. -DNOTCH_USE_CBLAS bench_twospirals.cpp \
    -L/opt/OpenBLAS/lib -lopenblas -o bench_twospirals \
    || (echo "Notch build FAILED")


echo -n "Notch twospirals SGD 1000 iters: "
test -f ./bench_twospirals &&
time -p ( ./bench_twospirals > notch.log ) 2> notch.err \
    || (echo -n "FAILED " ; echo "# see notch.err" ) \
    && (echo -n "OK " ; tail -1 notch.log ;
        NOTCH_TIME=$(awk '/^real /{print $2;}' notch.err)
        echo "Notch twospirals SGD real time: " $NOTCH_TIME )

echo -n "Notch twospirals SGD+ADADELTA 1000 iters: "
test -f ./bench_twospirals &&
time -p ( ./bench_twospirals --adadelta > notch-adadelta.log ) 2> notch-adadelta.err \
    || (echo -n "FAILED " ; echo "# see notch-adadelta.err" ) \
    && (echo -n "OK " ; tail -1 notch-adadelta.log ;
        NOTCH_ADADELTA_TIME=$(awk '/^real /{print $2;}' notch-adadelta.err)
        echo "Notch twospirals SGD+ADADELTA real time: " $NOTCH_ADADELTA_TIME )

echo '```'
echo
