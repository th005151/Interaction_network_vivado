#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "parameters.h"

void myproject(
	       input_t     N[N_NODES_MAX][N_FEATURES],
	       input_t     E[N_EDGES_MAX][E_FEATURES],
               index_t     receivers[N_EDGES_MAX][1],
               index_t     senders[N_EDGES_MAX][1],
	       result_t    e[N_EDGES_MAX][1],
	       unsigned short &const_size_in,
	       unsigned short &const_size_out);
#endif
