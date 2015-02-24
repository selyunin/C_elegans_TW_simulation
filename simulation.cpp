#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <numeric> 
#include <functional>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <limits>

#include <boost/filesystem.hpp>
#include <boost/math/special_functions/erf.hpp>
//#include <boost/regex.hpp>
//#include <boost/foreach.hpp>
//#include <boost/lexical_cast.hpp>

#include "sim_constants.h"
#include "save_results.h"

bool nif(double V1, const char* op, double V2, double variance, std::default_random_engine *generator);

int main(int argc, char *argv[])
{
	//generate random seed for the simulation
	std::ifstream f("/dev/urandom");
	unsigned int seed;
	f.read(reinterpret_cast<char*>(&seed), sizeof(seed));
	std::default_random_engine generator;
	generator.seed(seed);

	//create A and B matrices for computing V_eq
	double *A, *B;
	A = (double *) calloc(NUM_NEURONS*NUM_NEURONS,sizeof(double));
	B = (double *) calloc(NUM_NEURONS,sizeof(double));

	double *W_GAP_temp, *W_SYN_temp;
	W_GAP_temp = (double *) calloc(NUM_NEURONS, sizeof(double));
	W_SYN_temp = (double *) calloc(NUM_NEURONS, sizeof(double));

	for( int i = 0; i<NUM_NEURONS; i++){
		for(int k=0;k<NUM_NEURONS;k++){
			W_GAP_temp[k] = (double) W_GAP[i][k];
			W_SYN_temp[k] = (double) W_SYN[i][k];
		}
		for(int j = 0; j<NUM_NEURONS; j++){
			if(i != j){
				A[i*NUM_NEURONS+j] = -Res[i]*W_GAP[i][j]*G_GAP;
			}
			else{
				std::transform(W_GAP_temp, W_GAP_temp + NUM_NEURONS, W_GAP_temp, \
							std::bind1st(std::multiplies<double>(),G_GAP));
				std::transform(W_SYN_temp, W_SYN_temp + NUM_NEURONS, W_SYN_temp, \
							std::bind1st(std::multiplies<double>(),0.5*G_SYN));
				A[i*NUM_NEURONS+j] = 1 + Res[i]*( \
						std::accumulate(W_GAP_temp, W_GAP_temp+NUM_NEURONS, 0.0) + \
						std::accumulate(W_SYN_temp, W_SYN_temp+NUM_NEURONS, 0.0) \
						);
			}
		}
		for(int k=0;k<NUM_NEURONS;k++){
			W_SYN_temp[k] = ( (double) W_SYN[i][k])* E_syn[k] * G_SYN * 0.5;
		}
		B[i] = V_LEAK + Res[i]*std::accumulate(W_SYN_temp, W_SYN_temp+NUM_NEURONS, 0.0);
	}

	double *I_stim;
	I_stim = (double *) calloc(NUM_NEURONS*SIM_STEPS,sizeof(double));

	for(int curr_step=0;curr_step<SIM_STEPS;curr_step++){
		if ( curr_step >= START_IDX && curr_step <= END_IDX){
			I_stim[SIM_STEPS*STIM_IDX + curr_step] = PULSE_STIM;
		}
	}

	double *t_sim;
	t_sim = (double *) calloc(SIM_STEPS, sizeof(double));

	double *V_sim;
	V_sim = (double *) calloc(NUM_NEURONS*SIM_STEPS,sizeof(double));
	for(int n=0;n<NUM_NEURONS;n++){
		V_sim[n*SIM_STEPS] = V_init[n];
	}

	double *V_sim_step;
	V_sim_step = (double *) calloc(NUM_NEURONS, sizeof(double));
	std::copy(V_init, V_init+NUM_NEURONS, V_sim_step);

	
	double I_stim_step;
	double I_leak_step;
	double I_gap_step;
	double I_syn_step;
	double *g_step;
	g_step = (double *) calloc(NUM_NEURONS, sizeof(double));
	double *w_gap_temp;
	w_gap_temp = (double *) calloc(NUM_NEURONS, sizeof(double));
	double deltaV[NUM_NEURONS] = {0};

	for(int t_step=0;t_step<SIM_STEPS;t_step++){
		for(int i=0;i<NUM_NEURONS;i++){
			I_stim_step = I_stim[i*SIM_STEPS+t_step];
			I_leak_step = (V_LEAK - V_sim_step[i])/Res[i];
			I_gap_step=0;
			I_syn_step=0;
			for (int j=0;j<NUM_NEURONS;j++){
				w_gap_temp[j] = (double) (W_GAP[i][j]);
				I_gap_step += G_GAP*w_gap_temp[j]*(V_sim_step[j] - V_sim_step[i]);
				g_step[j] = G_SYN / (1 + exp(K*(V_sim_step[j]- V_eq[j])/V_RANGE));
				I_syn_step += W_SYN[i][j]*g_step[j]*(E_syn[j] - V_sim_step[i]);
			}
			deltaV[i] = (I_leak_step + I_gap_step + I_syn_step + I_stim_step)/Cap[i]*DT;
		}
		for(int j=0;j<NUM_NEURONS;j++){
			V_sim_step[j] = V_sim_step[j] + deltaV[j];
			V_sim[j*SIM_STEPS+t_step] = V_sim_step[j];
		}
	}

	double *V_sim_nif;
	V_sim_nif = (double *) calloc(NUM_NEURONS*SIM_STEPS,sizeof(double));

	for(int n=0;n<NUM_NEURONS;n++){
		V_sim_nif[n*SIM_STEPS] = V_init[n];
	}
	double *V_sim_step_nif;
	V_sim_step_nif = (double *) calloc(NUM_NEURONS, sizeof(double));
	std::copy(V_init, V_init+NUM_NEURONS, V_sim_step_nif);
	double g_branch;

	for(int t_step=0;t_step<SIM_STEPS;t_step++){
		for(int i=0;i<NUM_NEURONS;i++){
			I_stim_step = I_stim[i*SIM_STEPS+t_step];
			I_leak_step = (V_LEAK - V_sim_step_nif[i])/Res[i];
			I_gap_step=0;
			I_syn_step=0;
			for (int j=0;j<NUM_NEURONS;j++){
				w_gap_temp[j] = (double) (W_GAP[i][j]);
				I_gap_step += G_GAP*w_gap_temp[j]*(V_sim_step_nif[j] - V_sim_step_nif[i]);
				g_step[j] = 0;
				for(int k=0; k<NUM_CHANNELS*W_SYN[i][j]; k++){
					if (nif( V_sim_step_nif[j] , "<=", V_eq[j], -K/V_RANGE/1e4, &generator)){
						g_branch = 0;
					} else{
						g_branch = G_SYN/NUM_CHANNELS;
					}
					g_step[j] += g_branch;
				}
				I_syn_step +=  g_step[j]*(E_syn[j] - V_sim_step_nif[i]);
			}
			deltaV[i] = (I_leak_step + I_gap_step + I_syn_step + I_stim_step)/Cap[i]*DT;
		}
		for(int j=0;j<NUM_NEURONS;j++){
			V_sim_step_nif[j] = V_sim_step_nif[j] + deltaV[j];
			V_sim_nif[j*SIM_STEPS+t_step] = V_sim_step_nif[j];
		}
	}

	save_results(V_sim, "V_sim_cpp");
	save_results(V_sim_nif, "V_sim_nif_cpp");

	/////////////////////////////////////////////////////////////
	free(A);
	free(B);
	free(W_GAP_temp); 
	free(W_SYN_temp);
	free(I_stim);
	free(t_sim);
	free(V_sim);
	free(V_sim_step);
	free(g_step);
	free(w_gap_temp);

    return EXIT_SUCCESS;
}

bool nif(double V1, const char* op, double V2, double variance, std::default_random_engine *generator){
	std::normal_distribution<double> distribution(0, variance);
	double delta = 0;
	double sample;
	double eps=1e-20;
	if(strcmp(op,">") == 0){
		delta = V1 - V2 - eps;
	}
	else if(strcmp(op,">=") == 0){
		delta = V1 - V2;
	}
	else if(strcmp(op,"<") == 0){
		delta = V2 - V1 - eps;
	}
	else if(strcmp(op,"<=") == 0){
		delta = V2 - V1;
	} 
	else{
		std::cout<<"WRONG OPERATOR!"<<std::endl;
	}
	double	CDF =
				0.5 * (1 + erf(delta/sqrt(2.* pow(variance,2))))
				- eps;
	double alpha = (1. - CDF)/2.;
	if(alpha < 1e-16)
		alpha = 1e-16;
	double left_quantile =
				sqrt(2*pow(variance,2))*boost::math::erf_inv(2 * alpha - 1);
	double right_quantile =
				sqrt(2*pow(variance,2))*boost::math::erf_inv(2 * (1-alpha) - 1);
	sample = distribution(*generator);
	if (sample >= left_quantile && sample <= right_quantile){
		return true;
	}
	else{
		return false;
	}	
}

