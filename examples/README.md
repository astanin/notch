Examples
========

Notch examples can be founds in the `examples/` directory of the source
distribution:

 * `demo_xor` creates a multilayer peceptron and trains it to calculate
    XOR function (to classify two linearly inseparable sets).

 * `demo_iris` trains a multilayer perceptron to classify the iris flower
    dataset. The demo uses some optional Notch features:
    it reads data from a CSV file, applies one-hot encoding to labels,
    and applies softmax activation with cross-entropy loss on output.

 * `demo_mnist` trains a multilayer perceptron to classify hand-written
    digit images from the famous MNIST bechmark.

 * `demo_twospirals` trains a multilayer perceptron on a two-spiral
    classification problem. The problem is known to be difficult for
    neural networks. This example uses SQUARE dataset augmentation
    and ADADELTA training schedule to solve the problem.

 * `demo_io` shows how to save and load a multilayer perceptron.


How to build examples
---------------------

### On Linux

    mkdir build
    cd build
    cmake ..
    make


### On Windows

To build using GNU C++, install MinGW and CMake:

    mkdir build
    cd build
    cmake.exe -G "MinGW Makefiles" .`
    mingw32-make

For a debug build use

    cmake.exe -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug ..

To build using Visual Studio 2013 (Community):

    cmake.exe -G "Visual Studio 12 2013" ..

then open the solution.

### Linking with CBLAS

To enable CBLAS linking in examples, pass `-DUSE_CBLAS=YES` to cmake.

If CBLAS is installed in non-standard location, you may need to put it
in the CMAKE_PREFIX_PATH. For example, if you use Windows and installed
OpenBLAS to `C:\opt\OpenBLAS`, then you have to additionally pass
`-DCMAKE_PREFIX_PATH=c:/opt/OpenBLAS` to cmake.

### Using OpenMP

To build examples with OpenMP support (parallel computation),
pass `-DUSE_OPENMP=YES` to cmake.

If you use OpenMP and BLAS together, make sure that your BLAS library is
compatible with OpenMP (OpenBLAS should be compiled with OpenMP support).
