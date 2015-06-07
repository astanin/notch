NOTCH
=====

A C++11 implementation of the selected Neural Network algorithms.

This is (supposed to be)

 * a header-only C++11 library

 * without extra dependencies (CBLAS dependency is optional)

 * cross-platform (Linux and Windows)

 * reasonably fast (for a CPU-only implementations, HOPEFULLY)

 * but not at the cost of algorithms' readability and portability

This library is named after Notch, a transmembrane protein which acts as a
receptor and has a role, amonth other things, in neuronal function and
development.


How to use
----------

See `examples/`.


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


Bibliography
------------

 * NNLM3: Neural Networks and Learning Machines / Simon Haykin. -- 3rd ed.
   ([libgen](http://libgen.org/book/index.php?md5=0239f16656e6e5e7db7aaa160cf9f854))

 * PRML: Pattern Recognition and Machine Learning / Christoper M. Bishop. --
   ([libgen](http://libgen.org/book/index.php?md5=44807de3f3da5ae8f5d7066317d8a38a),
    [web](http://research.microsoft.com/en-us/um/people/cmbishop/prml/index.htm))

 * NNTT: Neural Networks: Tricks of the Trade / Grégoire Montavon et al (eds.) -- 2nd ed.
   ([libgen](http://libgen.org/book/index.php?md5=6b8768e619756f4e867282cfcec63f2e))

