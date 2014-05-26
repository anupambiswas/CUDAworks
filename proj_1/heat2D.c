#include<stdio.h>
#include<cuda.h>

// problem macros
#define N 200 // number spatial divisions in each x and y dimensions
#define dx 0.001 // spatial-step
#define tol 1.0e-8 // tolerance value for accuracy level
#define omega 1.0 // successive over-relaxtion factor
#define conv_check_CPU 100 // interval for checking convergence in CPU
#define conv_check_GPU 100 // interval for checking convergence in GPU
#define MAXIT 100000 // maximum number of iterations

// CUDA implementation macros
#define BSIZE 32 // block dimension size
#define numB 7 // number of blocks

// important: numB = minimum of value of I where I * BSIZE > N

typedef struct timespec tsp;

int Np1=N+1,Np2=N+2,Np2sq=(N+2)*(N+2),Np1np2=(N+1)*(N+2),Nnp2=N*(N+2);

__constant__ int np1[1],np2[1],np1np2[1],nnp2[1];

// resetBoundary() for CPU and GPU adjust the values of the ghost cells before every iteration

void resetBoundaryCPU(double *T)
{
  int i,j,jind;
  double x,y;
  for(i=1;i<Np1;i++)
    {
      x=(i-0.5)*dx;
      T[i]=2.0*sin(x)-T[i+Np2];
      T[i+Np1np2]=2.0*cos(x)-T[i+Nnp2];
      j=i;
      y=(j-0.5)*dx;
      jind=j*Np2;
      T[jind]=2*sin(y)-T[1+jind];
      T[Np1+jind]=2*cos(y)-T[N+jind];
    }
}

__global__
void resetBoundaryGPU(double *T)
{
  int ij=blockIdx.x*BSIZE+threadIdx.x+1;
  int jind;
  double xy;

  if(ij>N)
    return;
  xy=(ij-0.5)*dx;
  T[ij]=2.0*sin(xy)-T[ij+*np2];
  T[ij+*np1np2]=2.0*cos(xy)-T[ij+*nnp2];
  jind=ij**np2;
  T[jind]=2*sin(xy)-T[1+jind];
  T[*np1+jind]=2*cos(xy)-T[N+jind];
}

// calculation of heat source term

void calcSource(double *S)
{
  int i,j;
  double x,y;
  double hs=0.5*N*dx;
  double fac=20;
  for(i=1;i<Np1;i++)
    for(j=1;j<Np1;j++)
      {
	x=(i-0.5)*dx;
	y=(j-0.5)*dx;
	S[i+j*Np2]=sin(x)*cos(y)*exp(-fac*(fabs(x-hs)+fabs(y-hs))/hs);
      }
}

// iterate() for CPU and GPU gives the implementation of a single iteration step
// solver scheme is 2nd order
// solution method is Jacobi iteration with succesive over relaxation

void iterateCPU(double *T,double *Tn,double *S)
{
  int i,j,ind;
  double Tval;
  for(i=1;i<Np1;i++)
    for(j=1;j<Np1;j++)
      {
	ind=i+j*Np2;
	Tval=0.25*(T[ind-1]+T[ind+1]+T[ind-Np2]+T[ind+Np2])-S[ind];
	Tn[ind]=T[ind]+omega*(Tval-T[ind]);
      }
}

__global__ 
void iterateGPU(double *T,double *Tn,double *S)
{
  double Tval;
  int ind,i,j;
  i=blockIdx.x*BSIZE+threadIdx.x+1;
  j=blockIdx.y*BSIZE+threadIdx.y+1;
  if(i>N || j>N)
    return;
  ind=i+j**np2;
  Tval=0.25*(T[ind-1]+T[ind+1]+T[ind-*np2]+T[ind+*np2])-S[ind];
  Tn[ind]=T[ind]+omega*(Tval-T[ind]);
}

int convergedCPU(int iter,double *T,double *Tn)
{
  int i,j,ind;
  double maxErr=-1e40,err;
  for(i=1;i<Np1;i++)
    for(j=1;j<Np1;j++)
      {
	ind=i+j*Np2;
	err=fabs(Tn[ind]-T[ind]);
	if(err>maxErr)
	  maxErr=err;
      }
  printf("CPU iteration: %d %e\n",iter,maxErr);
  if(maxErr<tol)
    return(1);
  return(0);
}

// converged() functions are to determine if the solution has reached convergence

__global__
void convergedGPU(double *T,double *Tn,double *mxe)
{
  int i,j,jcon,jid,ind,tid;
  double err;
  __shared__ double mxErr[BSIZE];
  tid=threadIdx.x;
  mxErr[tid]=-1.0e40;
  __syncthreads();
  i=blockIdx.x*BSIZE+threadIdx.x+1;
  jcon=blockIdx.y*BSIZE+1;
  if(i>N)
    return;
  for(j=0;j<BSIZE;j++)
    {
      jid=jcon+j;
      if(jid>N)
	continue;
      ind=i+jid**np2;
      err=fabs(T[ind]-Tn[ind]);
      if(err>mxErr[tid])
	mxErr[tid]=err;
    }
  __syncthreads();
  int stride;
  for(stride=BSIZE/2;stride>0;stride/=2)
    {
      if(tid<stride)
	mxErr[tid]=mxErr[tid]>mxErr[tid+stride]?mxErr[tid]:mxErr[tid+stride];
      __syncthreads();
    }
  mxe[blockIdx.x+blockIdx.y*numB]=mxErr[0];
}

int convergedGPU(int iter,double *errGPU)
{
  int i;
  double max=-1.0e40;
  int tnb=numB*numB;
  for(i=0;i<tnb;i++)
    if(max<errGPU[i])
      max=errGPU[i];
  printf("GPU iteration: %d %e\n",iter,max);
  if(max<tol)
    return(1);
  return(0);
}

// initializer function

void initialize(double *var,double *val,int num)
{
  int i;
  for(i=0;i<num;i++)
    var[i]=num;
}

// function to calculate time of computation

double computeTime(tsp t0,tsp t1)
{
  tsp dif;
  dif.tv_sec=t1.tv_sec-t0.tv_sec;
  dif.tv_nsec=t1.tv_nsec-t0.tv_nsec;
  if(dif.tv_nsec<0)
    {
      dif.tv_nsec+=1000000000;
      dif.tv_sec--;
    }
  return(dif.tv_sec+dif.tv_nsec/1.0e9);
}

// function to store data in files

void printData(double *T,char fname[])
{
  FILE *F;
  int i,j;
  F=fopen(fname,"w");
  for(i=1;i<Np1;i++)
    for(j=1;j<Np1;j++)
      fprintf(F,"%e %e %e\n",(i-0.5)*dx,(j-0.5)*dx,T[i+j*Np2]);
  fclose(F);
  printf("\n");
}

int main(int argc,char *argv[])
{
  double *TH,*TD,*SH,*SD;
  double *TnH,*TnD;
  double *mxErr,*mxErrc;
  tsp ths,the,tds,tde;
  int numBsq=numB*numB;
  int maxit;

  maxit=MAXIT;
  if(argc>1)
    maxit=atoi(argv[1]);

  cudaMemcpyToSymbol(np1,&Np1,sizeof(int));
  cudaMemcpyToSymbol(np2,&Np2,sizeof(int));
  cudaMemcpyToSymbol(np1np2,&Np1np2,sizeof(int));
  cudaMemcpyToSymbol(nnp2,&Nnp2,sizeof(int));

  // allocating memory on CPU and GPU
  TH=(double*)malloc(sizeof(double)*Np2sq);
  TnH=(double*)malloc(sizeof(double)*Np2sq);
  SH=(double*)malloc(sizeof(double)*Np2sq);
  mxErrc=(double*)malloc(sizeof(double)*numBsq);
  cudaMalloc((void**)&TD,sizeof(double)*Np2sq);
  cudaMalloc((void**)&TnD,sizeof(double)*Np2sq);
  cudaMalloc((void**)&SD,sizeof(double)*Np2sq);
  cudaMalloc((void**)&mxErr,sizeof(double)*numBsq);

  int iter;

  // variale initialization and memory copying
  initialize(TH,0,Np2sq);
  initialize(TnH,0,Np2sq);
  calcSource(SH);
  cudaMemcpy(TD,TH,sizeof(double)*Np2sq,cudaMemcpyHostToDevice);
  cudaMemcpy(TnD,TnH,sizeof(double)*Np2sq,cudaMemcpyHostToDevice);
  cudaMemcpy(SD,SH,sizeof(double)*Np2sq,cudaMemcpyHostToDevice);

  // CPU computation
  iter=0;
  clock_gettime(CLOCK_REALTIME,&ths);
  while(1)
    {
      resetBoundaryCPU(TH);
      iterateCPU(TH,TnH,SH);
      resetBoundaryCPU(TnH);
      iterateCPU(TnH,TH,SH);
      if(iter%conv_check_CPU==0)
	{
	  if(convergedCPU(iter,TH,TnH))
	    break;
	  if(iter>maxit)
	    break;
	}
      iter++;
    }
  clock_gettime(CLOCK_REALTIME,&the);
  printData(TH,"T_CPU");

  // GPU computation
  dim3 dimG(numB,numB,1),dimB(BSIZE,BSIZE,1);
  dim3 dimG2(numB,1,1),dimB2(BSIZE,1,1);
  iter=0;
  clock_gettime(CLOCK_REALTIME,&tds);
  while(1)
    {
      resetBoundaryGPU<<<dimG2,dimB2>>>(TD);
      iterateGPU<<<dimG,dimB>>>(TD,TnD,SD);
      resetBoundaryGPU<<<dimG2,dimB2>>>(TnD);
      iterateGPU<<<dimG,dimB>>>(TnD,TD,SD);
      if(iter%conv_check_GPU==0)
      	{
      	  convergedGPU<<<dimG,dimB2>>>(TD,TnD,mxErr);
      	  cudaMemcpy(mxErrc,mxErr,sizeof(double)*numBsq,cudaMemcpyDeviceToHost);
      	  if(convergedGPU(iter,mxErrc))
      	    break;
	  if(iter>maxit)
	    break;
      	}
      iter++;
    }
  clock_gettime(CLOCK_REALTIME,&tde);
  cudaMemcpy(TH,TD,sizeof(double)*Np2sq,cudaMemcpyDeviceToHost);
  printData(TH,"T_GPU");

  // outputting computation time for CPU and GPU
  double timH,timD;
  timH=computeTime(ths,the);
  timD=computeTime(tds,tde);
  printf("\nCPU computation time: %e seconds\n\n",timH);
  printf("\nGPU computation time: %e seconds\n\n",timD);
  printf("\nGPU speed gain: %e\n\n",timH/timD);

  // freeing memory from CPU and GPU
  free(TH);
  free(TnH);
  free(SH);
  free(mxErrc);
  cudaFree(TD);
  cudaFree(TnD);
  cudaFree(SD);
  cudaFree(mxErr);
  return(0);
}
