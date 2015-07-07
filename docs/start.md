Getting Started
===============

@mainpage

Notch is a header-only C++11 library which allows to implement
feed-forward neural networks.

## Library Structure

The library consists of several C++ include files.

 * `notch.hpp` (`notch::core` namespace) is the only "required" file.
   It implements all the "core" functionality of the neural networks.

 * `notch_io.hpp` (`notch::io` namespace)
   defines basic facilities to save and load data and neural network
   configuration.

 * `notch_pre.hpp` (`notch::pre` namespace)
   offers some basic facilities to preprocess (prepare) data before feeding
   it to the neural network.

 * `notch_metrics.hpp` (`notch::metrics` namespace)
   allows to estimate the "quality" of the trained neural network.

The users are encouraged to use other libraries for advanced preprocessing,
input-output and visualization needs.

## Dependencies

The library tries to get as far as possible while relying only on the
standard C++ library. It should be usable without additional dependencies.

Most user-facing APIs operate on the array data type (`std::valarray`)
from the standard library, so it should be easy to integrate Notch into
other programs. The array type is referred to as `Array` throughout the
Notch sources and documentation.

### Optional Dependencies

You may link Notch programs with a BLAS (Basic Linear Algebra Subprograms)
library, such as OpenBLAS or ATLAS. It allows to improve Notch
performance where BLAS is available.
BLAS interoperation is disabled by default unless you `#define NOTCH_USE_CBLAS`
macro before including Notch headers.

You may enable OpenMP parallel code execution in Notch. This may improve
Notch performance on multicore CPUs.
OpenMP is disabled by default. It requires the compiler to support OpenMP
standard (GCC is very good, MSVC doesn't support the latest OpenMP version).
To enable OpenMP, `#define NOTCH_USE_OPENMP` and use an appropriate
compiler flag (`-fopenmp` on GCC, both for compiler and the linker).

## Installation

Get the most recent source from Git repository:

~~~
git clone https://astanin@bitbucket.org/astanin/notch.git
~~~

From your project simply `#include "notch.hpp"` and other Notch files.
You may have to set your compiler include path to let it find the Notch
headers (see `-I` flag documentation if you're using GCC or Clang).

If you include Notch headers in more than one compilation unit (.cpp file),
then make sure to `#define NOTCH_ONLY_DECLARATIONS` _before_ the includes
in all but one of them to avoid multiple definitions.

## The Basics

Most small programs use at least `notch.hpp` and `notch_io.hpp`
headers. This is a common beginning of a small Notch program:

~~~{.cpp}
#include <iostream>
#include "notch.hpp"     // notch::core
#include "notch_io.hpp"  // notch::io

using namespace notch;
using namespace std;

int main() {
    // your code goes here
}
~~~

Neural networks are represented by a `Net` class. It is a stack of several
neural layers which support backpropagation (all subclasses of
`ABackpropLayer`) and a loss layer (see `ALossLayer`).  The easiest way to
construct new `Net` objects is to use `MakeNet` class:

~~~{.cpp}
// Create a network with 2 inputs, 4 hidden nodes and 1 output
// with an Euclidean (L2) loss layer, and initialize it randomly.
auto net = MakeNet()
    .setInputDim(2) // 2 inputs
    .addFC(4)       // then a fully connected layer with 4 nodes
    .addL2Loss()    // Euclidean loss
    .init();        // or .make() to skip initialization
~~~

To apply the neural network to some input, use `.output()` method of
the `Net` object.  It takes an `Array` and returns another `Array`.
(Remember, Notch `Array`s are just `std::valarray<float>`).

To train a network in supervised mode, use `SGD::train()` method. 
It implements Stochastic Gradient Descent, a first-order online
learning algorithm. It will require to choose `ALearningPolicy`
 and a labeled dataset for training (`LabeledDataset`).

The labeled dataset can be loaded from a CSV file
(see `CSVReader` class and `demo_iris.cpp` example),
from a labeled set of images in IDX format
(see `IDXReader` class and `demo_mnist.cpp` example),
or constructed in code using `.append()` method
of a `LabeledDataset` object.

~~~{.cpp}
// Assuming that the label to learn is the last
// column on a CSV table, we may read training data like this:
auto trainingSet = CSVReader("my-training-data.csv").read();

// ADADELTA is a good adaptive learning policy
net.setLearningPolicy(AdaDelta());

// Train for 100 epochs:
SGD::train(net, trainingSet, 100);
~~~

To save the trained network to file, use
`PlainTextNetworkWriter` class. To load parameters of the
previously trained network, use `PlainTextNetworkReader`.
Both are defined in `notch_io.hpp`:

~~~{.cpp}
PlainTextNetworkWriter(std::cout) << net;
~~~
