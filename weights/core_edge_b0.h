//Numpy array shape [8]
//Min -0.439332400893
//Max 0.820810661459
//Number of zeros 0

#ifndef CORE_EDGE_B0_H_
#define CORE_EDGE_B0_H_

#ifndef __SYNTHESIS__
ap_fixed<16,6> core_edge_b0[8];
#else
ap_fixed<16,6> core_edge_b0[8] = {0.211806, -0.439332, 0.205264, -0.189313, 0.188628, -0.032380, -0.094698, 0.820811};
#endif

#endif
