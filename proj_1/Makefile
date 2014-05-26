heat: heat2D.cu
	nvcc -arch=sm_30 heat2D.cu -lrt -o heat
heat2D.cu: heat2D.c
	cp heat2D.c heat2D.cu
clean:
	rm -f *~
resetf:
	rm -f *~ heat heat2D.cu T_*