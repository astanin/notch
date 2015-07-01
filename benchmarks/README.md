Notch vs Torch7
===============

Benchmark date:   2015-07-01T15:38+0000

Benchmark CPU:    Intel(R) Core(TM) i7-2600 CPU @ 3.40GHz

Benchmark OS:     Linux 3.16.0-30-generic

C++ compiler:     g++ (Ubuntu 4.8.4-2ubuntu1~14.04) 4.8.4

BLAS:             libopenblas.so


Benchmarks
----------

 * twospirals: a 2-50-50-50-50-10-1 multilayer perceptron


Log
---

```
Torch7 twospirals SGD 1000 iters: OK [0m# training error = 0.99146835152189[0m	
Torch7 twospirals SGD real time:  107.23
Notch twospirals SGD 1000 iters: OK # training error = 0.923686
Notch twospirals SGD real time:  121.75
Notch twospirals SGD+ADADELTA 1000 iters: OK # training error = 0.973669
Notch twospirals SGD+ADADELTA real time:  115.00
```

