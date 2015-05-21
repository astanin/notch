NOTCH
=====

A C++11 implementation of the selected Neural Network algorithms.

This is (supposed to be)

 * a header-only C++11 library

 * without extra dependencies

 * cross-platform (Linux and Windows)

 * reasonably fast (for a CPU-only implementations, HOPEFULLY)

 * but not at the cost of algorithms' readability and portability

This library is named after Notch, a transmembrane protein which acts as a
receptor and has a role, amonth other things, in neuronal function and
development.


How to use
----------

See `examples/'.


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

Install MinGW and CMake:

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

To build with Visual Studio 2013

```
cmake.exe -G "" ..
```

Bibliography
------------

 * NNLM3: Neural networks and learning machines / Simon Haykin. -- 3rd ed.
   ([libgen][libgen], [amazon.uk][amazon])


[nnlm3-libgen]: http://libgen.org/book/index.php?md5=0239f16656e6e5e7db7aaa160cf9f854
[nnlm3-amazon]: http://www.amazon.co.uk/Neural-Networks-Learning-Machines-3rd/dp/8120340000/

