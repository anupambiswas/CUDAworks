This project deals with a 2D heat transfer problem for square domain. The code studies the computation speeds for CPU and GPU.

The side of the square domain is divided in N parts. The length of each such part is given by dx (set equal to 0.001 as a macro).

The boundary conditions are given by trigonometric functions. A source term for each spatial position is also taken into account.

Observation: It is noticed that as the N increases, the speed gain of GPU over CPU increases. For N = 200, the speed gain is about 16. For N = 500, speed gain is over 60. For N = 1000, the speed gain is almost 100.

Running guidelines:

$ make
# This produces the executable 'heat'

$ ./heat
# This starts the simulation

To visualize the plots, gnuplot can be used

$ gnuplot
gnuplot> p 'T_CPU' w image

In a different tab,
$ gnuplot
gnuplot> p 'T_GPU' w image
