@mainpage

NOTCH
=====

Feed-forward Neural Networks in C++11 for the rest of us.

This is (supposed to be)

 * a header-only C++11 library

 * without extra dependencies (CBLAS dependency is optional)

 * cross-platform (Linux and Windows)

 * reasonably fast (for a CPU-only implementations, HOPEFULLY)

 * but not at the cost of algorithms' readability and portability

This library is named after Notch, a transmembrane protein which acts as a
receptor and has a role, amonth other things, in neuronal function and
development.

Motivation
----------

 * Most of the neural network frameworks are notoriously difficult to
   install and deploy.

   Some of them work only on a particular operating system flavor,
   or have very specific hardware requirements.

   Notch is supposed to lower the barrier to entry and be a tool which
   works anywhere where a modern C++ compiler is available.
   Just copy a header file. No need to install anything.

 * Many neural network frameworks are designed to _train_ neural
   networks. Few care about _using_ them and integrating with end-user
   software.

   Notch is designed to be embedded into other software.


How to use
----------

See `examples/`.


Conditional Compilation
-----------------------

Set this compilation flags _before_ including "notch.hpp" and other Notch headers:

    #define NOTCH_USE_CBLAS

if you're linking your program with a CBLAS library and want to use an efficient
implemtation of the linear algebra.

    #define NOTCH_USE_OPENMP

to parallelize Notch computations using OpenMP.
If you use OpenMP and BLAS together, make sure that your BLAS library is
compatible with OpenMP (OpenBLAS should be compiled with OpenMP support).

    #define NOTCH_ONLY_DECLARATIONS

If you include Notch headers in more than one compilation unit (source file),
then, to suppress multiple definitions, `#define NOTCH_ONLY_DECLARATIONS`
before Notch includes in all but one of the source files.
See how Notch is used in `test/` source file to see an example.

    #define NOTCH_DISABLE_OPTIMIZATIONS

to disable hand-written optimizations (if you suspect there is a bug).


Naming Conventions
------------------

Different neural network libraries name the same things differently.
This library sticks to terminology of NNLM3.

For neurons with activation potential $v_j = W_{ji} y_i + b_j$,
activation function $\phi(v_j)$, and the network loss $E$, the naming
convention is summarized in the following table:

    | CONCEPT                      | NOTCH                 | TORCH7 NN           | CAFFE          |

    | network layer                | ABackpropLayer        | Module              | Layer          |
    | loss E calculator            | ALossLayer            | Criterion           | LossLayer      |
    | entire network               | Net                   | Sequential          | Net            |
    | network initializer          | Init                  | ???                 | Filler         |
    | network optimizer            | SGD                   | StochasticGradient  | SGDSolver      |

    | forward propagation step     | .output()             | :forward()          | .Forward()     |
    | back propagation step        | .backprop()           | :backward()         | .Backward()    |
    | update weights               | .update()             | :updateParameters() | .Update()      |
    | read weights W_{ji}          | .getWeights()         | .weight             | .layer_param() |
    | read bias    b_j             | .getBias()            | .bias               | .layer_param() |

    | \partial E / \partial W_{ji} | weight sensitivity    | .gradWeight         | ???            |
    | \partial E / \partial b_j    | bias sensitivity      | .gradBias           | ???            |
    | \partial E / \partial y_i    | error signals, errors | .gradInput          | ???            |
    | \partial E / \partial v_j    | local gradient        | .gradOutput         | ???            |


Bibliography
------------

 * NNLM3: Neural Networks and Learning Machines / Simon Haykin. -- 3rd ed.
   ([libgen](http://libgen.org/book/index.php?md5=0239f16656e6e5e7db7aaa160cf9f854))

 * PRML: Pattern Recognition and Machine Learning / Christoper M. Bishop. --
   ([libgen](http://libgen.org/book/index.php?md5=44807de3f3da5ae8f5d7066317d8a38a),
    [web](http://research.microsoft.com/en-us/um/people/cmbishop/prml/index.htm))

 * NNTT: Neural Networks: Tricks of the Trade / Grégoire Montavon et al (eds.) -- 2nd ed.
   ([libgen](http://libgen.org/book/index.php?md5=6b8768e619756f4e867282cfcec63f2e))

