Notch vs Torch7
===============

Until Notch gets more features and more rigorous benchmarks are written, 
this is a quick-and-dirty comparison of Notch (this library) and Torch7
(a popular and one of the best performing deep learning frameworks).

The libraries are compared in CPU mode. Torch7 is installed using 
ezinstall script from http://torch.ch/docs/getting-started.html
Notch implementation logs the same amount of information as the Torch7 script.
Output is redirected to files.

Notch version:    b360d99a

Torch7 version:   b30f4bd3


Benchmark date:   2015-07-07T16:12+0000

Benchmark CPU:    Intel(R) Core(TM) i7-2600 CPU @ 3.40GHz

Benchmark OS:     Ubuntu 14.04.2 LTS, Trusty Tahr ;  Linux x86_64

C++ compiler:     g++ (Ubuntu 4.8.4-2ubuntu1~14.04) 4.8.4

BLAS:             libopenblas_sandybridgep-r0.2.14.so


Benchmarks
----------

### Two Spirals

Run 1000 epochs of training a 2-50-50-50-50-10-1 multilayer perceptron.


|library and task                                  |   time, s| samples/s|
|:-------------------------------------------------|---------:|---------:|
|Torch7 SGD FixedRate                              |     23.37|      8258|
|Notch SGD FixedRate                               |     11.70|     16496|
|Notch SGD ADADELTA                                |     11.12|     17356|
|Notch (OpenMP, no BLAS) SGD FixedRate             |     15.17|     12722|
|Notch (no OpenMP, no BLAS) SGD FixedRate          |     11.05|     17466|

