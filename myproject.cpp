#include <iostream>
#include "parameters.h"
#include "myproject.h"

#include "nnet_dense.h"
#include "nnet_activation.h"
#include "nnet_dense_large.h"
#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_graph.h"

//insert weights from training
#include "weights/core_edge_w0.h"
#include "weights/core_edge_b0.h"
#include "weights/core_edge_w1.h"
#include "weights/core_edge_b1.h"
#include "weights/core_edge_w2.h"
#include "weights/core_edge_b2.h"
#include "weights/core_edge_w3.h"
#include "weights/core_edge_b3.h"
#include "weights/core_node_w0.h"
#include "weights/core_node_b0.h"
#include "weights/core_node_w1.h"
#include "weights/core_node_b1.h"
#include "weights/core_node_w2.h"
#include "weights/core_node_b2.h"

void myproject(
	       input_t      N[N_NODES_MAX][N_FEATURES],
	       input_t      E[N_EDGES_MAX][E_FEATURES],
               index_t      receivers[N_EDGES_MAX][1],
               index_t      senders[N_EDGES_MAX][1],
	       result_t     e[N_EDGES_MAX][1],
	       unsigned short &const_size_in,
	       unsigned short &const_size_out)
{

  //hls-fpga-machine-learning insert IO
#pragma HLS ARRAY_RESHAPE variable=N complete dim=2
#pragma HLS ARRAY_RESHAPE variable=E complete dim=2
#pragma HLS ARRAY_RESHAPE variable=receivers complete dim=2
#pragma HLS ARRAY_RESHAPE variable=senders complete dim=2
#pragma HLS ARRAY_RESHAPE variable=e complete dim=2
#pragma HLS INTERFACE ap_vld port=N,E,receivers,senders,e


  const_size_in	= N_NODES_MAX*N_FEATURES+N_EDGES_MAX*E_FEATURES+2*N_EDGES_MAX*1;
  const_size_out = N_EDGES_MAX*1;

#ifndef __SYNTHESIS__
  static bool loaded_weights = false;
 if (!loaded_weights) {
   //hls-fpga-machine-learning insert load weights                                                                           
   nnet::load_weights_from_txt<model_default_t, E_FEATURES*LATENT_EDGE + 2*N_FEATURES*LATENT_EDGE>(core_edge_w0, "core_edge_w0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_EDGE>(core_edge_b0, "core_edge_b0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_EDGE*LATENT_EDGE>(core_edge_w1, "core_edge_w1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_EDGE>(core_edge_b1, "core_edge_b1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_EDGE*LATENT_EDGE>(core_edge_w2, "core_edge_w2.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_EDGE>(core_edge_b2, "core_edge_b2.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_EDGE*E_FEATURES>(core_edge_w3, "core_edge_w3.txt");
   nnet::load_weights_from_txt<model_default_t, E_FEATURES>(core_edge_b3, "core_edge_b3.txt");
   nnet::load_weights_from_txt<model_default_t, E_FEATURES*LATENT_NODE + N_FEATURES*LATENT_NODE>(core_node_w0, "core_node_w0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_NODE>(core_node_b0, "core_node_b0.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_NODE*LATENT_NODE>(core_node_w1, "core_node_w1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_NODE>(core_node_b1, "core_node_b1.txt");
   nnet::load_weights_from_txt<model_default_t, LATENT_NODE*N_FEATURES>(core_node_w2, "core_node_w2.txt");
   nnet::load_weights_from_txt<model_default_t, N_FEATURES>(core_node_b2, "core_node_b2.txt");

   loaded_weights = true;
 }
#endif

  //interaction network
  input_t effects[N_EDGES_MAX][E_FEATURES];
  input_t aggregation[N_NODES_MAX][E_FEATURES];
  input_t influence[N_NODES_MAX][N_FEATURES];
  input_t q[N_NODES_MAX][E_FEATURES];

  //Use BRAM for pipeline
  #pragma HLS ARRAY_PARTITION variable=effects complete dim=2
  #pragma HLS ARRAY_PARTITION variable=aggregation complete dim=2

  //Use BRAM for pipeline
  #pragma HLS ARRAY_PARTITION variable=influence complete dim=2
  #pragma HLS ARRAY_PARTITION variable=q complete dim=2
  
  
  //DO_PRAGMA(HLS PIPELINE II=REUSE_GRAPH)

  //edge block
  nnet::relational_model<input_t, index_t, input_t, graph_config1>(E, N, receivers, senders, effects, aggregation, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  //node block
  nnet::object_model<input_t, input_t, graph_config2>(N, aggregation, influence, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
  //edge block
  nnet::relational_model<input_t, index_t, input_t, graph_config3>(effects, influence, receivers, senders, e, q, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);

}
