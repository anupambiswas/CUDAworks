#include<cuda.h>
#include<stdlib.h>
#include<math.h>
#include<stdio.h>

#define BSIZE 64 // block-thread size; number 64 chosen to reduce warp divergence

// macros to implement constant value boundary conditions (Dirichlet)
#define bcMulLeft (-1.0)
#define bcMulRight (-1.0)
#define bcAddLeft 2.0
#define bcAddRight 0.0

typedef struct timespec tsp;

// GPU single iteration step: central difference scheme
__global__ void iterateGPU(double *T,double *Tn,int N)
{
  int index=blockIdx.x*BSIZE+threadIdx.x+1;
  if(index>N)
    return;
  Tn[index]=0.5*(T[index-1]+T[index+1]);
}

// GPU function to reset the ghost values to implement Dirichlet boundary conditions
__global__ void resetBoundaryGPU(double *T,int N)
{
  T[0]=bcMulLeft*T[1]+bcAddLeft;
  T[N+1]=bcMulRight*T[N]+bcAddRight;
}

// CPU single iteration step: central difference scheme
void iterateCPU(double *T,double *Tn,int N)
{
  int i;
  for(i=1;i<=N;i++)
    Tn[i]=0.5*(T[i-1]+T[i+1]);
}

// CPU function to reset the ghost values to implement Dirichlet boundary conditions
void resetBoundaryCPU(double *T,int N)
{
  T[0]=bcMulLeft*T[1]+bcAddLeft;
  T[N+1]=bcMulRight*T[N]+bcAddRight;
}

void computeTime(tsp t0,tsp t1,char *msg=NULL);

// function to record time intervals
void computeTime(tsp t0,tsp t1,char *msg)
{
  tsp dif;
  dif.tv_sec=t1.tv_sec-t0.tv_sec;
  dif.tv_nsec=t1.tv_nsec-t0.tv_nsec;
  if(dif.tv_nsec<0)
    {
      dif.tv_nsec+=1000000000;
      dif.tv_sec--;
    }
  if(msg==NULL)
    {
      char mssg[]="Total computation time";
      msg=mssg;
    }
  printf("\n%s: %f seconds\n\n",msg,dif.tv_sec+dif.tv_nsec*1.0e-9);
}

int main(int argc,char *argv[])
{
  int i;
  double *TH,*TD; // temperature (host and device)
  double dx; // spatial-step
  double *eH,*eD; // error variables (host and device)
  int N; // mesh size
  int sz; // memory requirement
  FILE *F;
  int NOI; // number of iterations
  int iter; // iterating variable
  tsp ths,the,tds,tde; // time variables for CPU and GPU computing

  N=1000;
  NOI=10000;

  if(argc>1)
    N=atoi(argv[1]); // command-line input to set N
  if(argc>2)
    NOI=atoi(argv[2]); // command-line input to set NOI 

  int NOIh=NOI/2; // half of total number of iteration, as in a single loop two iterations will be carried out

  dx=1.0/N;
  sz=sizeof(double)*(N+2);

  // storing exact solutions in file
  double x;
  F=fopen("T_exact","w");
  for(i=0;i<1000;i++)
    {
      x=i/999.0;
      fprintf(F,"%e %e\n",x,1-x);
    }
  fclose(F);

  //computing on CPU
  TH=(double*)malloc(sz);
  TD=(double*)malloc(sz);
  for(i=0;i<N+2;i++)
    TH[i]=TD[i]=0.0;
  clock_gettime(CLOCK_REALTIME,&ths);
  for(iter=0;iter<NOIh;iter++)
    {
      if(iter%1000==0)
	printf("CPU iteration: %d\n",2*iter);

      resetBoundaryCPU(TH,N);
      iterateCPU(TH,TD,N);

      resetBoundaryCPU(TD,N);
      iterateCPU(TD,TH,N);
    }
  clock_gettime(CLOCK_REALTIME,&the);
  F=fopen("T_CPU","w");
  for(i=1;i<=N;i++)
    fprintf(F,"%e %e\n",(i-0.5)*dx,TH[i]);
  fclose(F);
  free(TH);
  free(TD);
  printf("\n");

  // memory allocation - CPU
  TH=(double*)malloc(sz);
  eH=(double*)malloc(sz);

  // memory allocation - GPU
  cudaMalloc((void**)&TD,sz);
  cudaMalloc((void**)&eD,sz);

  // initializing temperature and error on host
  for(i=1;i<=N;i++)
    {
      eH[i]=TH[i]=0.0;
    }

  // copying memory from CPU to GPU
  cudaMemcpy(TD,TH,sz,cudaMemcpyHostToDevice);
  cudaMemcpy(eD,eH,sz,cudaMemcpyHostToDevice);

  // main solution iteration loop
  dim3 dimG((N+1)/BSIZE+1,1,1),dimB(BSIZE,1,1);
  clock_gettime(CLOCK_REALTIME,&tds);
  for(iter=0;iter<NOIh;iter++)
    {
      if(iter%1000==0)
	printf("GPU iteration: %d\n",2*iter);

      resetBoundaryGPU<<<1,1>>>(TD,N);
      iterateGPU<<<dimG,dimB>>>(TD,eD,N);

      resetBoundaryGPU<<<1,1>>>(eD,N);
      iterateGPU<<<dimG,dimB>>>(eD,TD,N);
    }
  clock_gettime(CLOCK_REALTIME,&tde);

  cudaMemcpy(TH,TD,sz,cudaMemcpyDeviceToHost);
  F=fopen("T_GPU","w");
  for(i=1;i<=N;i++)
    fprintf(F,"%e %e\n",(i-0.5)*dx,TH[i]);
  fclose(F);

  // freeing memory - CPU
  free(TH);
  free(eH);

  // freeing memory - GPU
  cudaFree(TD);
  cudaFree(eD);

  // result details printing
  computeTime(ths,the,"CPU computation time");
  computeTime(tds,tde,"GPU computation time");

  printf("%d %e %e\n",N,dx,N*dx);

  return(0);
}
