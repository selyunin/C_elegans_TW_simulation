#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#include <iostream>
#include <numeric> 
#include <functional>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <limits>

#include "sim_constants.h"
#include "save_results.h"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__constant__ int d_W_GAP[NUM_NEURONS][NUM_NEURONS];
__constant__ int d_W_SYN[NUM_NEURONS][NUM_NEURONS];
__constant__ double d_Cap[NUM_NEURONS];
__constant__ double d_Res[NUM_NEURONS];
__constant__ double d_E_syn[NUM_NEURONS];
__constant__ double d_V_init[NUM_NEURONS];
__constant__ double d_V_eq[NUM_NEURONS];
#define N 9


__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
} 

__global__ void integrate(double* V_sim,
						  double* V_sim_step,
						  double* deltaV,
						  double* I_stim,
						  double* g_step,
						  double* w_gap_temp)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	double I_stim_step;
	double I_leak_step;
	double I_gap_step;
	double I_syn_step;
	for(int t_step=0;t_step<SIM_STEPS;t_step++){
		I_stim_step = I_stim[i*SIM_STEPS+t_step];
		I_leak_step = (V_LEAK - V_sim_step[i])/d_Res[i];
		I_gap_step=0;
		I_syn_step=0;
		for (int j=0;j<NUM_NEURONS;j++){
			w_gap_temp[j] = (double) (d_W_GAP[i][j]);
			I_gap_step += G_GAP*w_gap_temp[j]*(V_sim_step[j] - V_sim_step[i]);
			g_step[j] = G_SYN / (1 + exp(K*(V_sim_step[j]- d_V_eq[j])/V_RANGE));
			I_syn_step += d_W_SYN[i][j]*g_step[j]*(d_E_syn[j] - V_sim_step[i]);
		}
		deltaV[i] = (I_leak_step + I_gap_step + I_syn_step + I_stim_step)/d_Cap[i]*DT;
		V_sim_step[i] = V_sim_step[i] + deltaV[i];
		V_sim[i*SIM_STEPS+t_step] = V_sim_step[i];
		__syncthreads();
	}
}
__device__ bool nif(double V1,
				    const char* op, 
				    double V2, 
				    double variance,
				    curandState* globalState,
				    int idx)

{
	double delta = 0;
	double sample;
	double eps=1e-20;
	//printf("thread[%d]: V1 = %f\n",idx,V1);
	//printf("thread[%d]: V2 = %f\n",idx,V2);
	delta = V2 - V1;
	//printf("thread[%d]: delta = %f\n",idx,delta);
	double	CDF =
				0.5 * (1 + erf(delta/sqrt(2.* pow(variance,2.))))
				- eps;
	//printf("thread[%d]: CDF = %f\n",idx,CDF);
	double alpha = (1. - CDF)/2.;
	//printf("thread[%d]: aplha = %f\n",idx,alpha);
	if(alpha < 1e-14)
		alpha = 1e-14;
	double left_quantile =
				sqrt(2*pow(variance,2.))*erfinv(2 * alpha - 1);
	double right_quantile =
				sqrt(2*pow(variance,2.))*erfinv(2 * (1-alpha) - 1);
	//printf("thread[%d]: l_q = %f\n",idx,left_quantile);
	//printf("thread[%d]: r_q = %f\n",idx,right_quantile);
    sample = variance*curand_normal(globalState+idx);
	//printf("thread[%d]: sample = %f\n",idx,sample);

	if (sample >= left_quantile && sample <= right_quantile){
		//printf("thread[%d]: result = true\n",idx);
		return true;
	}
	else{
		//printf("thread[%d]: result = false\n",idx);
		return false;
	}	
}

__global__ void integrate_nif(double* V_sim,
						 	  double* V_sim_step,
						 	  double* deltaV,
						 	  double* I_stim,
						 	  double* g_step,
						 	  double* w_gap_temp,
						 	  curandState* globalState)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	double I_stim_step;
	double I_leak_step;
	double I_gap_step;
	double I_syn_step;
	double g_branch;
	for(int t_step=0;t_step<SIM_STEPS;t_step++){
		I_stim_step = I_stim[i*SIM_STEPS+t_step];
		I_leak_step = (V_LEAK - V_sim_step[i])/d_Res[i];
		I_gap_step=0;
		I_syn_step=0;
		for (int j=0;j<NUM_NEURONS;j++){
			w_gap_temp[j*i+j] = (double) (d_W_GAP[i][j]);
			I_gap_step += G_GAP*w_gap_temp[j*i+j]*(V_sim_step[j] - V_sim_step[i]);
			g_step[j*i+j] = 0;
			for(int k=0; k<1*d_W_SYN[i][j]; k++){
				if (nif( V_sim_step[j] , "<=", d_V_eq[j], -K/V_RANGE/1.e4, globalState,i )){
					g_branch = 0.;
				} else{
					g_branch = G_SYN/1.;
				}
				g_step[j*i+j] += g_branch;
			}
			I_syn_step += g_step[i*j+j]*(d_E_syn[j] - V_sim_step[i]);
		}
		deltaV[i] = (I_leak_step + I_gap_step + I_syn_step + I_stim_step)/d_Cap[i]*DT;
		V_sim_step[i] = V_sim_step[i] + deltaV[i];
		V_sim[i*SIM_STEPS+t_step] = V_sim_step[i];
		__syncthreads();
	}
}


int main(int argc, char *argv[])
{
    //generate random seed for the simulation
	std::ifstream f("/dev/urandom");
	unsigned int seed;
	f.read(reinterpret_cast<char*>(&seed), sizeof(seed));

/////////////////////
	double *I_stim;
	I_stim = (double *) calloc(NUM_NEURONS*SIM_STEPS,sizeof(double));

	for(int curr_step=0;curr_step<SIM_STEPS;curr_step++){
		if ( (curr_step >= START_IDX) && (curr_step <= END_IDX)){
			I_stim[SIM_STEPS*STIM_IDX + curr_step] = PULSE_STIM;
		}
	}
	double *V_sim;
	V_sim = (double *) calloc(NUM_NEURONS*SIM_STEPS,sizeof(double));
	double *V_sim_nif;
	V_sim_nif = (double *) calloc(NUM_NEURONS*SIM_STEPS,sizeof(double));

	for(int n=0;n<NUM_NEURONS;n++){
		V_sim[n*SIM_STEPS] = V_init[n];
	}
	double *V_sim_step;
	V_sim_step = (double *) calloc(NUM_NEURONS, sizeof(double));
	std::copy(V_init, V_init+NUM_NEURONS, V_sim_step);

	double *g_step;
	g_step = (double *) calloc(NUM_NEURONS, sizeof(double));
	double *w_gap_temp;
	w_gap_temp = (double *) calloc(NUM_NEURONS, sizeof(double));
	double deltaV[NUM_NEURONS] = {0};
//allocate data on the device
	double *d_I_stim;
	CUDA_CALL(cudaMalloc((double **)&d_I_stim, NUM_NEURONS*SIM_STEPS*sizeof(double)));
	CUDA_CALL(cudaMemcpy(d_I_stim, I_stim, NUM_NEURONS*SIM_STEPS*sizeof(double),
				cudaMemcpyHostToDevice));
	double *d_V_sim;
	CUDA_CALL(cudaMalloc((double **)&d_V_sim, NUM_NEURONS*SIM_STEPS*sizeof(double)));
	double *d_V_sim_step;
	CUDA_CALL(cudaMalloc((double **)&d_V_sim_step, NUM_NEURONS*sizeof(double)));
	CUDA_CALL(cudaMemcpy(d_V_sim_step, V_sim_step, NUM_NEURONS*sizeof(double),
				cudaMemcpyHostToDevice));

	double *d_V_sim_nif;
	CUDA_CALL(cudaMalloc((double **)&d_V_sim_nif, NUM_NEURONS*SIM_STEPS*sizeof(double)));
	double *d_V_sim_step_nif;
	CUDA_CALL(cudaMalloc((double **)&d_V_sim_step_nif, NUM_NEURONS*sizeof(double)));
	CUDA_CALL(cudaMemcpy(d_V_sim_step_nif, V_sim_step, NUM_NEURONS*sizeof(double),
				cudaMemcpyHostToDevice));

	double *d_g_step;
	CUDA_CALL(cudaMalloc((double **)&d_g_step, NUM_NEURONS*NUM_NEURONS*sizeof(double)));
	CUDA_CALL(cudaMemcpy(d_g_step, g_step, NUM_NEURONS*NUM_NEURONS*sizeof(double),
				cudaMemcpyHostToDevice));
	double *d_w_gap_temp;
	CUDA_CALL(cudaMalloc((double **)&d_w_gap_temp, NUM_NEURONS*NUM_NEURONS*sizeof(double)));
	CUDA_CALL(cudaMemcpy(d_w_gap_temp, w_gap_temp, NUM_NEURONS*NUM_NEURONS*sizeof(double),
				cudaMemcpyHostToDevice));
	double *d_deltaV;
	CUDA_CALL(cudaMalloc((double **)&d_deltaV, NUM_NEURONS*sizeof(double)));
	CUDA_CALL(cudaMemcpy(d_deltaV, deltaV, NUM_NEURONS*sizeof(double),
				cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_W_GAP, W_GAP, NUM_NEURONS*NUM_NEURONS*sizeof(int)));
	CUDA_CALL(cudaMemcpyToSymbol(d_W_SYN, W_SYN, NUM_NEURONS*NUM_NEURONS*sizeof(int)));
	CUDA_CALL(cudaMemcpyToSymbol(d_Cap, Cap,NUM_NEURONS*sizeof(double)));
	CUDA_CALL(cudaMemcpyToSymbol(d_Res,Res,NUM_NEURONS*sizeof(double)));
	CUDA_CALL(cudaMemcpyToSymbol(d_E_syn,E_syn,NUM_NEURONS*sizeof(double)));
	CUDA_CALL(cudaMemcpyToSymbol(d_V_init,V_init,NUM_NEURONS*sizeof(double)));
	CUDA_CALL(cudaMemcpyToSymbol(d_V_eq,V_eq,NUM_NEURONS*sizeof(double)));

	//allocation done
    // setup random generator
    curandState* d_genState;
	dim3 tpb(NUM_NEURONS,1,1);
    cudaMalloc ( &d_genState, NUM_NEURONS*sizeof( curandState ) );
    setup_kernel <<< 1, tpb >>> ( d_genState, seed );

	integrate<<< 1, 9 >>>(d_V_sim,
						  d_V_sim_step,
						  d_deltaV,
						  d_I_stim,
						  d_g_step,
						  d_w_gap_temp);

	integrate_nif<<< 1, 9 >>>(d_V_sim_nif,
							  d_V_sim_step_nif,
						      d_deltaV,
							  d_I_stim,
							  d_g_step,
							  d_w_gap_temp,
							  d_genState);
	//cudaThreadSynchronize();
	CUDA_CALL(cudaMemcpy(V_sim, d_V_sim, NUM_NEURONS*SIM_STEPS*sizeof(double),
				cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(V_sim_nif, d_V_sim_nif, NUM_NEURONS*SIM_STEPS*sizeof(double),
				cudaMemcpyDeviceToHost));

	save_results(V_sim, "V_sim_cu");
	save_results(V_sim_nif, "V_sim_nif_cu");

	CUDA_CALL(cudaFree(d_I_stim));
	CUDA_CALL(cudaFree(d_V_sim));
	CUDA_CALL(cudaFree(d_V_sim_step));
	CUDA_CALL(cudaFree(d_V_sim_step_nif));
	CUDA_CALL(cudaFree(d_V_sim_nif));
	CUDA_CALL(cudaFree(d_g_step));
	CUDA_CALL(cudaFree(d_w_gap_temp));
	CUDA_CALL(cudaFree(d_deltaV));

    return EXIT_SUCCESS;
}

