#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <iostream>

std::vector<double> tbsla::mpi::Matrix::page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations){
	int proc_rank; 
	MPI_Comm_rank(comm, &proc_rank); 
	std::vector<double> b(n_col, 1.0);
	bool converge = false; 
	int nb_iterations = 0;
	std::vector<double> b_t;
	double max, error, teleportation_sum;

	while(!converge && nb_iterations <= max_iterations){
		b_t = b; 
		
		b = this->spmv(comm, b_t); 
		// std::cout << b_t.size() << std::endl; 
		MPI_Allreduce(b.data(), &max, b.size(), MPI_DOUBLE, MPI_MAX, comm); 
		MPI_Allreduce(b_t.data(), &teleportation_sum, n_col, MPI_DOUBLE, MPI_SUM, comm);
		 
		teleportation_sum *= (1-beta)/n_col; 

		for(int  i = 0 ; i < n_col; i++){
			b[i] = beta*b[i] + teleportation_sum; 
		}
		error = 0.0; 
		for(int i = 0; i < n_col; i++){
			b[i] = b[i]/max; 
			error += std::abs(b[i] - b_t[i]);  
		}
		if(error < epsilon)
			converge = true;
		nb_iterations++;  
	}
	return b; 
}