#include "nnet_graph.h"
  void relational_model_dataflow(
	input_t    E[N_NODES_MAX][N_FEATURES],
	input_t    N[N_EDGES_MAX][E_FEATURES],
	index_t   receivers[N_EDGES_MAX][1],
	index_t   senders[N_EDGES_MAX][1],
	input_t     effects_tmp[4][10][N_FEATURES],
	input_t     aggregation_tmp[4][N_NODES_MAX][E_FEATURES],
	model_default_t core_edge_w0[E_FEATURES*LATENT_EDGE + 2*N_FEATURES*LATENT_EDGE],
	model_default_t core_edge_b0[LATENT_EDGE],
	model_default_t core_edge_w1[LATENT_EDGE*LATENT_EDGE],
	model_default_t core_edge_b1[LATENT_EDGE],
	model_default_t core_edge_w2[LATENT_EDGE*LATENT_EDGE],
	model_default_t core_edge_b2[LATENT_EDGE],
	model_default_t core_edge_w3[LATENT_EDGE*E_FEATURES],
	model_default_t core_edge_b3[E_FEATURES])
  {
    #pragma HLS DATAFLOW 
    nnet::relational_model<input_t, index_t, input_t, graph_config1>(0, 10,E, N, receivers, senders, effects_tmp[0], aggregation_tmp[0], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
    nnet::relational_model<input_t, index_t, input_t, graph_config1>(10,20,E, N, receivers, senders, effects_tmp[1], aggregation_tmp[1], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
    nnet::relational_model<input_t, index_t, input_t, graph_config1>(20,30,E, N, receivers, senders, effects_tmp[2], aggregation_tmp[2], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
    nnet::relational_model<input_t, index_t, input_t, graph_config1>(30,40,E, N, receivers, senders, effects_tmp[3], aggregation_tmp[3], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  }