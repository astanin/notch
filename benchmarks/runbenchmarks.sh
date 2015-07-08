#!/bin/bash

BENCHMARK_DATE=$(date --iso-8601=minutes -u)
CPU_MODEL=$(cat /proc/cpuinfo | grep '^model name' | head -1 | sed 's/[^:]*: //')
OS_INFO=$(. /etc/os-release && echo $NAME $VERSION '; ' $(uname -sm))
BLAS_LIBRARY=$(ls -S /opt/OpenBLAS/lib/libopenblas*[0-9]*.so | head -1 | sed 's/.*\///')
CXX=${CXX:=g++}

NOTCH_VERSION=$(git rev-parse HEAD|fold -8|head -1)
TORCH7_VERSION=$((cd ~/torch/pkg/torch && git rev-parse HEAD)|fold -8|head -1)


N_ITERS=1000
N_SAMPLES=193  # twospirals-train


# TODO: write Notch benchmarks for https://github.com/soumith/convnet-benchmarks

echo "Notch vs Torch7"
echo "==============="
echo
echo "Until Notch gets more features and more rigorous benchmarks are written, "
echo "this is a quick-and-dirty comparison of Notch (this library) and Torch7"
echo "(a popular and one of the best performing deep learning frameworks)."
echo
echo "The libraries are compared in CPU mode. Torch7 is installed using "
echo "ezinstall script from http://torch.ch/docs/getting-started.html"
echo "Notch implementation logs the same amount of information as the Torch7 script."
echo "Output is redirected to files."
echo

echo "Notch version:    $NOTCH_VERSION" ; echo
echo "Torch7 version:   $TORCH7_VERSION" ; echo
echo

echo "Benchmark date:   $BENCHMARK_DATE" ; echo
echo "Benchmark CPU:    $CPU_MODEL" ; echo
echo "Benchmark OS:     $OS_INFO" ; echo
echo "C++ compiler:     $($CXX --version | head -1)" ; echo
echo "BLAS:             ${BLAS_LIBRARY}" ; echo
echo

echo "Benchmarks"
echo "----------"
echo
echo "### Two Spirals"
echo
echo "Run $N_ITERS epochs of training a 2-50-50-50-50-10-1 multilayer perceptron."
echo

rm *.err *.log
RESULTS=./benchmark_results.txt
test -f $RESULTS && rm $RESULTS
>$RESULTS  printf "|%-50s|%10s|%10s|\n" "library and task" "time, s" "samples/s"
>>$RESULTS printf "|:-------------------------------------------------|---------:|---------:|\n"

export LC_NUMERIC=C
export LD_LIBRARY_PATH=/opt/OpenBLAS/lib

(
    # requires torch7 (th) to be installed
    which th > /dev/null
    HAS_TORCH=$?

    if [[ $HAS_TORCH -eq 0 ]]; then
        echo -n "Torch SGD..." ;
        time -p ( th bench_twospirals.lua $N_ITERS > torch.log ) 2> torch.err \
            && ( echo OK ; \
                 TORCH_TIME=$(awk '/^real /{print $2;}' torch.err) ; \
                 TORCH_SPS=$(awk "BEGIN {print $N_ITERS * $N_SAMPLES / $TORCH_TIME;}") ; \
                 >> $RESULTS printf "|%-50s|%10.2f|%10.0f|\n" \
                             "Torch7 SGD FixedRate" $TORCH_TIME $TORCH_SPS) \
            || ( echo "FAILED # see torch.err" )
    fi
) >&2

(
    # requires GNU C++ compiler and OpenBLAS in /opt/OpenBLAS
    test -f ./bench_twospirals_blas && rm ./bench_twospirals_blas
    ${CXX} -std=c++11 -O3 -I.. -DNOTCH_USE_CBLAS -I/opt/OpenBLAS/include \
        bench_twospirals.cpp \
        -L/opt/OpenBLAS/lib -lopenblas \
        -o bench_twospirals_blas \
        || (echo "Notch build with BLAS FAILED")
    HAS_NOTCH=$(test -f ./bench_twospirals_blas)

    if [[ $HAS_NOTCH -eq 0 ]]; then
        echo -n "Notch SGD... " ;
        time -p ( ./bench_twospirals_blas $N_ITERS > notch_sgd.log ) 2> notch_sgd.err \
            && ( echo OK ; \
                 EXE_TIME=$(awk '/^real /{print $2;}' notch_sgd.err) ; \
                 EXE_SPS=$(awk "BEGIN {print $N_ITERS * $N_SAMPLES / $EXE_TIME;}") ; \
                 >> $RESULTS printf "|%-50s|%10.2f|%10.0f|\n" \
                             "Notch SGD FixedRate" $EXE_TIME $EXE_SPS) \
            || ( echo "FAILED # see notch_sgd.err" )
    fi

    if [[ $HAS_NOTCH -eq 0 ]]; then
        echo -n "Notch ADADELTA... " ;
        time -p ( ./bench_twospirals_blas $N_ITERS --adadelta > notch_adadelta.log ) 2> notch_adadelta.err  \
            && ( echo OK ; \
                 EXE_TIME=$(awk '/^real /{print $2;}' notch_adadelta.err ) ; \
                 EXE_SPS=$(awk "BEGIN {print $N_ITERS * $N_SAMPLES / $EXE_TIME;}") ; \
                 >> $RESULTS printf "|%-50s|%10.2f|%10.0f|\n" \
                             "Notch SGD ADADELTA" $EXE_TIME $EXE_SPS) \
            || ( echo "FAILED # see notch_adadelta.err" )
    fi
) >&2

(
    # requires only GNU C++ compiler
    test -f ./bench_twospirals_noblas && rm ./bench_twospirals_noblas
    ${CXX} -std=c++11 -O3 -I.. -fopenmp -DNOTCH_USE_OPENMP \
        bench_twospirals.cpp -o bench_twospirals_noblas \
        || (echo "Notch build without BLAS FAILED")
    HAS_NOTCH=$(test -f ./bench_twospirals_noblas)

    if [[ $HAS_NOTCH -eq 0 ]]; then
        echo -n "Notch SGD FixedRate (no BLAS)..." ;
        time -p ( ./bench_twospirals_noblas $N_ITERS > notch_noblas_sgd.log ) 2> notch_noblas_sgd.err \
            && ( echo OK ; \
                 EXE_TIME=$(awk '/^real /{print $2;}' notch_noblas_sgd.err) ; \
                 EXE_SPS=$(awk "BEGIN {print $N_ITERS * $N_SAMPLES / $EXE_TIME;}") ; \
                 >> $RESULTS printf "|%-50s|%10.2f|%10.0f|\n" \
                             "Notch (OpenMP, no BLAS) SGD FixedRate" $EXE_TIME $EXE_SPS) \
            || ( echo "FAILED # see notch_noblas_sgd.err" )
    fi

) >&2

(
    # requires only GNU C++ compiler
    test -f ./bench_twospirals_baseline && rm ./bench_twospirals_baseline
    ${CXX} -std=c++11 -O3 -I.. \
        bench_twospirals.cpp -o bench_twospirals_baseline \
        || (echo "Notch build without BLAS FAILED")
    HAS_NOTCH=$(test -f ./bench_twospirals_baseline)

    if [[ $HAS_NOTCH -eq 0 ]]; then
        echo -n "Notch SGD FixedRate (no BLAS, no OpenMP)..." ;
        time -p ( ./bench_twospirals_baseline $N_ITERS > notch_baseline_sgd.log ) 2> notch_baseline_sgd.err \
            && ( echo OK ; \
                 EXE_TIME=$(awk '/^real /{print $2;}' notch_baseline_sgd.err) ; \
                 EXE_SPS=$(awk "BEGIN {print $N_ITERS * $N_SAMPLES / $EXE_TIME;}") ; \
                 >> $RESULTS printf "|%-50s|%10.2f|%10.0f|\n" \
                             "Notch (no OpenMP, no BLAS) SGD FixedRate" $EXE_TIME $EXE_SPS) \
            || ( echo "FAILED # see notch_baseline_sgd.err" )
    fi

) >&2

echo
cat $RESULTS
echo
