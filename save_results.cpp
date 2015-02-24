#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "sim_constants.h"
#include "save_results.h"

void save_results(double* Vdata, const char* filename){
	std::ofstream output_file;
	output_file.open( filename );
	for(int t=0;t<SIM_STEPS ;t++){
		output_file<<t*DT;
		for(int n=0;n<NUM_NEURONS;n++){
			output_file<<"\t"<<Vdata[n*SIM_STEPS+t];
		}
		output_file<<std::endl;
	}
	output_file.close();
}
