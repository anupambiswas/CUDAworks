heat: heat.cu
	nvcc heat.cu -arch=sm_30 -lrt -o heat
heat.cu: heat.c
	cp heat.c heat.cu
clean:
	rm -f *~
resetf:
	rm -f *~ heat T_exact T_GPU T_CPU