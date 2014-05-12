This project is about solving a non-dimensional steady-state heat equation in a single-dimensional domain of unit length. Dirichlet boundary conditions are used, that is, T(x = 0) = 1 and T(x = 1) = 0.

The goal of the project is not to reach the converged solution but to compare the speeds of computation on CPU and GPU for the same extents of computation.

Computation scheme:
Jacobi iteration

Two important variables:
N = number of divisions of the domain, default value 1000
NOI = number of iterations, default value 10000

Jacobi iteration is not the best iteration scheme and usually takes a large number of steps to reach convergence. Default values of N and NOI do not end in convergence. If N = 1000, NOI must be about 1000000 for convergence.

Also, if the domain length is kept constant (here unity) and N is increased, it takes relatively more number of steps to converge. Hence if N is increased, NOI must also be increased to obtain converged results.

If one is just concerned about comparing computation speeds, NOI is not important.

Building and running instructions:

$ make
# creates the executable 'heat'

$ ./heat
# runs the execuatble with default values N = 1000, NOI = 10000
# these values do not give converged solutions though

$ ./heat 10000
# runs the executable with N = 10000

$ ./heat 10000 1000000
# runs the executable with N = 10000 and NOI = 1000000

Result files:
The CPU results are written in T_CPU and GPU files in T_GPU. In order to have a visual idea of extent of solution progress, one can compare these results with T_exact which has exact solutions.

If gnuplot is installed:

$ gnuplot
gnuplot> p 'T_CPU' w l,'T_GPU' w l,'./T_exact' w l

If python is installed (with matplotlib), one can plot each file individually from the CLI:

$ python dataPlot.py T_CPU &
$ python dataPlot.py T_GPU &
$ python dataPlot.py T_exact &
