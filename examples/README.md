Notch Examples
==============

 * `demo_xor` creates a multilayer peceptron and trains it to calculate
    XOR function (to classify two linearly inseparable sets)

 * `demo_iris` trains a multilayer perceptron to classify the iris flower
    dataset; the demo uses some optional Notch features:
    it reads data from a CSV file, applies one-hot encoding to labels,
    and applies softmax activation with cross-entropy loss on output

 * `demo_io` shows how to save and load a multilayer perceptron


How to build examples
---------------------

### On Linux

```
mkdir build
cd build
cmake ..
make
```


### On Windows

To build using GNU C++, install MinGW and CMake:

```
mkdir build
cd build
cmake.exe -G "MinGW Makefiles" .`
mingw32-make
```

For a debug build use

```
cmake.exe -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
```

To build using Visual Studio 2013 (Community):

```
cmake.exe -G "Visual Studio 12 2013" ..
```

then open the solution.

### Linking with CBLAS

To enable CBLAS linking in examples, pass `-DUSE_CBLAS=YES` to cmake.

