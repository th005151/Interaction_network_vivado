#include <iostream>
#include "parameters.h"
#include "myproject.h"


#include "nnet_dense.h"
#include "nnet_activation.h"
#include "nnet_dense_large.h"
#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_graph.h"
#include "nnet_merge.h"

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


	void relational_dataflow_1(
  input_t      N[N_NODES_MAX][N_FEATURES],
  input_t      E[N_EDGES_MAX][E_FEATURES],
  index_t      receivers[N_EDGES_MAX][1],
  index_t      senders[N_EDGES_MAX][1],
  input_t     effects_tmp[4][N_EDGES_MAX/4][E_FEATURES],
  input_t     aggregation_tmp[4][N_NODES_MAX][E_FEATURES]
){
  #pragma HLS DATAFLOW
  // input_t      N_rep[4][N_NODES_MAX][N_FEATURES];
  // input_t      E_rep[4][N_EDGES_MAX][E_FEATURES];
  // index_t      receivers_rep[4][N_EDGES_MAX][1];
  // index_t      senders_rep[4][N_EDGES_MAX][1];

  // #pragma HLS STREAM variable =N_rep
  // #pragma HLS STREAM variable =E_rep
  // #pragma HLS STREAM variable =receivers_rep
  // #pragma HLS STREAM variable =senders_rep

  // nnet::replicate<input_t, N_NODES_MAX, N_FEATURES>(N, N_rep[0], N_rep[1], N_rep[2], N_rep[3]);
  // nnet::replicate<input_t, N_EDGES_MAX, E_FEATURES>(E, E_rep[0], E_rep[1], E_rep[2], E_rep[3]);
  // nnet::replicate<index_t, N_EDGES_MAX, 1>(receivers, receivers_rep[0], receivers_rep[1], receivers_rep[2], receivers_rep[3]);
  // nnet::replicate<index_t, N_EDGES_MAX, 1>(senders, senders_rep[0], senders_rep[1], senders_rep[2], senders_rep[3]);

  // nnet::relational_model<input_t, index_t, input_t, graph_config1>(0,0*N_EDGES_MAX/4,1*N_EDGES_MAX/4,E_rep[0], N_rep[0], receivers_rep[0], senders_rep[0], effects_tmp[0], aggregation_tmp[0], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  // nnet::relational_model<input_t, index_t, input_t, graph_config1>(1,1*N_EDGES_MAX/4,2*N_EDGES_MAX/4,E_rep[1], N_rep[1], receivers_rep[1], senders_rep[1], effects_tmp[1], aggregation_tmp[1], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  // nnet::relational_model<input_t, index_t, input_t, graph_config1>(2,2*N_EDGES_MAX/4,3*N_EDGES_MAX/4,E_rep[2], N_rep[2], receivers_rep[2], senders_rep[2], effects_tmp[2], aggregation_tmp[2], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  // nnet::relational_model<input_t, index_t, input_t, graph_config1>(3,3*N_EDGES_MAX/4,4*N_EDGES_MAX/4,E_rep[3], N_rep[3], receivers_rep[3], senders_rep[3], effects_tmp[3], aggregation_tmp[3], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  nnet::relational_model<input_t, index_t, input_t, graph_config1>(0,0*N_EDGES_MAX/4,1*N_EDGES_MAX/4,E, N, receivers, senders, effects_tmp[0], aggregation_tmp[0], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  nnet::relational_model<input_t, index_t, input_t, graph_config1>(1,1*N_EDGES_MAX/4,2*N_EDGES_MAX/4,E, N, receivers, senders, effects_tmp[1], aggregation_tmp[1], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  nnet::relational_model<input_t, index_t, input_t, graph_config1>(2,2*N_EDGES_MAX/4,3*N_EDGES_MAX/4,E, N, receivers, senders, effects_tmp[2], aggregation_tmp[2], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  nnet::relational_model<input_t, index_t, input_t, graph_config1>(3,3*N_EDGES_MAX/4,4*N_EDGES_MAX/4,E, N, receivers, senders, effects_tmp[3], aggregation_tmp[3], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);

};
// void relational_dataflow_2(
//   input_t      N[N_NODES_MAX][N_FEATURES],
//   input_t      E[N_EDGES_MAX][E_FEATURES],
//   index_t      receivers[N_EDGES_MAX][1],
//   index_t      senders[N_EDGES_MAX][1],
//   input_t     effects_tmp[4][10][E_FEATURES],
//   input_t     aggregation_tmp[4][N_NODES_MAX][E_FEATURES],
// ){
//   #pragma HLS DATAFLOW
//   nnet::relational_model<input_t, index_t, input_t, graph_config3>(0,0, 10,E, N, receivers, senders, effects_tmp[0], aggregation_tmp[0], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
//   nnet::relational_model<input_t, index_t, input_t, graph_config3>(1,10,20,E, N, receivers, senders, effects_tmp[1], aggregation_tmp[1], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
//   nnet::relational_model<input_t, index_t, input_t, graph_config3>(2,20,30,E, N, receivers, senders, effects_tmp[2], aggregation_tmp[2], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
//   nnet::relational_model<input_t, index_t, input_t, graph_config3>(3,30,40,E, N, receivers, senders, effects_tmp[3], aggregation_tmp[3], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);

// };

// void relational_dataflow_2(
//   input_t effects[N_EDGES_MAX][E_FEATURES],
//   input_t influence[N_NODES_MAX][N_FEATURES],
//   index_t      receivers[N_EDGES_MAX][1],
//   index_t      senders[N_EDGES_MAX][1],
//   result_t e_tmp[4][N_EDGES_MAX/4][E_FEATURES],
//   input_t q_tmp[4][N_NODES_MAX][E_FEATURES]
// ){
//   //nnet::object_model<input_t, input_t, graph_config2>(0, 0, 28, N, aggregation, influence_tmp[0], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);

//   #pragma HLS DATAFLOW
//   nnet::relational_model_2d<input_t, index_t, input_t, graph_config3>(0,0*N_EDGES_MAX/4, 1*N_EDGES_MAX/4, effects, influence, receivers, senders, e_tmp[0], q_tmp[0], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
//   nnet::relational_model_2d<input_t, index_t, input_t, graph_config3>(1,1*N_EDGES_MAX/4, 2*N_EDGES_MAX/4, effects, influence, receivers, senders, e_tmp[1], q_tmp[1], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
//   nnet::relational_model_2d<input_t, index_t, input_t, graph_config3>(2,2*N_EDGES_MAX/4, 3*N_EDGES_MAX/4, effects, influence, receivers, senders, e_tmp[2], q_tmp[2], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
//   nnet::relational_model_2d<input_t, index_t, input_t, graph_config3>(3,3*N_EDGES_MAX/4, 4*N_EDGES_MAX/4, effects, influence, receivers, senders, e_tmp[3], q_tmp[3], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  
// };

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
#pragma HLS ARRAY_RESHAPE variable=N complete dim=0
#pragma HLS ARRAY_RESHAPE variable=E complete dim=0
#pragma HLS ARRAY_RESHAPE variable=receivers complete dim=0
#pragma HLS ARRAY_RESHAPE variable=senders complete dim=0
//#pragma HLS ARRAY_RESHAPE variable=e complete dim=0
#pragma HLS INTERFACE ap_vld port=N,E,receivers,senders,e


  // const_size_in	= N_NODES_MAX*N_FEATURES+N_EDGES_MAX*E_FEATURES+2*N_EDGES_MAX*1;
  // const_size_out = N_EDGES_MAX*1;

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
  #pragma HLS ARRAY_PARTITION variable=effects complete dim=0
  #pragma HLS ARRAY_PARTITION variable=aggregation complete dim=0

  //Use BRAM for pipeline
  #pragma HLS ARRAY_PARTITION variable=influence complete dim=0
  //#pragma HLS RESOURCE variable=influence core=RAM_S2P_LUTRAM 
  #pragma HLS ARRAY_PARTITION variable=q complete dim=0
  



  input_t effects_tmp[4][N_EDGES_MAX/4][E_FEATURES];
  input_t aggregation_tmp[4][N_NODES_MAX][E_FEATURES]={0};
  #pragma HLS ARRAY_PARTITION variable=effects_tmp complete dim=0

  #pragma HLS ARRAY_PARTITION variable=aggregation_tmp complete dim=0
  #pragma HLS ARRAY_PARTITION variable=aggregation_tmp complete dim=3


  input_t      N_rep[4][N_NODES_MAX][N_FEATURES];
  input_t      E_rep[4][N_EDGES_MAX][E_FEATURES];
  index_t      receivers_rep[4][N_EDGES_MAX][1];
  index_t      senders_rep[4][N_EDGES_MAX][1];

  //#pragma HLS DATAFLOW

  nnet::replicate<input_t, N_NODES_MAX, N_FEATURES>(N, N_rep[0], N_rep[1], N_rep[2], N_rep[3]);
  nnet::replicate<input_t, N_EDGES_MAX, E_FEATURES>(E, E_rep[0], E_rep[1], E_rep[2], E_rep[3]);
  nnet::replicate<index_t, N_EDGES_MAX, 1>(receivers, receivers_rep[0], receivers_rep[1], receivers_rep[2], receivers_rep[3]);
  nnet::replicate<index_t, N_EDGES_MAX, 1>(senders, senders_rep[0], senders_rep[1], senders_rep[2], senders_rep[3]);
  
  nnet::relational_model<input_t, index_t, input_t, graph_config1>(0,0*N_EDGES_MAX/4,1*N_EDGES_MAX/4,E_rep[0], N_rep[0], receivers_rep[0], senders_rep[0], effects_tmp[0], aggregation_tmp[0], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  nnet::relational_model<input_t, index_t, input_t, graph_config1>(1,1*N_EDGES_MAX/4,2*N_EDGES_MAX/4,E_rep[1], N_rep[1], receivers_rep[1], senders_rep[1], effects_tmp[1], aggregation_tmp[1], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  nnet::relational_model<input_t, index_t, input_t, graph_config1>(2,2*N_EDGES_MAX/4,3*N_EDGES_MAX/4,E_rep[2], N_rep[2], receivers_rep[2], senders_rep[2], effects_tmp[2], aggregation_tmp[2], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  nnet::relational_model<input_t, index_t, input_t, graph_config1>(3,3*N_EDGES_MAX/4,4*N_EDGES_MAX/4,E_rep[3], N_rep[3], receivers_rep[3], senders_rep[3], effects_tmp[3], aggregation_tmp[3], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);


  //relational_dataflow_1(N, E, receivers, senders, effects_tmp, aggregation_tmp);
  //nnet::merge_relation_agg<input_t, index_t, input_t, graph_config1>(effects_tmp, receivers, effects, aggregation);
  nnet::merge_relation<input_t, input_t, graph_config1>(effects_tmp, effects);
  nnet::aggregation<input_t, index_t, input_t, graph_config1>(effects, receivers, aggregation);
  
//--------------------------------------------------------------------------- 
  //object
  input_t influence_tmp[4][N_NODES_MAX/4][N_FEATURES];
  input_t aggregation_rep[4][N_NODES_MAX][E_FEATURES]={0};
  #pragma HLS ARRAY_PARTITION variable=influence_tmp complete dim=1
  #pragma HLS ARRAY_PARTITION variable=influence_tmp complete dim=3
  #pragma HLS ARRAY_PARTITION variable=aggregation_rep complete dim=1
  #pragma HLS ARRAY_PARTITION variable=aggregation_rep complete dim=3
  
  nnet::replicate<input_t, N_NODES_MAX, E_FEATURES>(aggregation, aggregation_rep[0], aggregation_rep[1], aggregation_rep[2], aggregation_rep[3]);

  nnet::object_model<input_t, input_t, graph_config2>(0, 0*N_NODES_MAX/4, 1*N_NODES_MAX/4, N_rep[0], aggregation_rep[0], influence_tmp[0], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
  nnet::object_model<input_t, input_t, graph_config2>(1, 1*N_NODES_MAX/4, 2*N_NODES_MAX/4, N_rep[1], aggregation_rep[1], influence_tmp[1], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
  nnet::object_model<input_t, input_t, graph_config2>(2, 2*N_NODES_MAX/4, 3*N_NODES_MAX/4, N_rep[2], aggregation_rep[2], influence_tmp[2], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
  nnet::object_model<input_t, input_t, graph_config2>(3, 3*N_NODES_MAX/4, 4*N_NODES_MAX/4, N_rep[3], aggregation_rep[3], influence_tmp[3], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
  
  nnet::merge_object<input_t, input_t, graph_config2>(influence_tmp, influence);

//--------------------------------------------------------------------------- 

  result_t e_tmp[4][N_EDGES_MAX/4][E_FEATURES];
  input_t q_tmp[4][N_NODES_MAX][E_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=e_tmp complete dim=1
  #pragma HLS ARRAY_PARTITION variable=e_tmp complete dim=3
  #pragma HLS ARRAY_PARTITION variable=q_tmp complete dim=0
  #pragma HLS ARRAY_PARTITION variable=q_tmp complete dim=3


  input_t effects_rep[4][N_EDGES_MAX][E_FEATURES];
  input_t influence_rep[4][N_NODES_MAX][N_FEATURES];
  #pragma HLS ARRAY_PARTITION variable=effects_rep complete dim=0
  #pragma HLS ARRAY_PARTITION variable=influence_rep complete dim=0

  nnet::replicate<input_t, N_EDGES_MAX, E_FEATURES>(effects, effects_rep[0], effects_rep[1], effects_rep[2], effects_rep[3]);
  nnet::replicate<input_t, N_NODES_MAX, N_FEATURES>(influence, influence_rep[0], influence_rep[1], influence_rep[2], influence_rep[3]);

  nnet::relational_model<input_t, index_t, input_t, graph_config3>(0,0*N_EDGES_MAX/4, 1*N_EDGES_MAX/4, effects_rep[0], influence_rep[0], receivers, senders, e_tmp[0], q_tmp[0], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  nnet::relational_model<input_t, index_t, input_t, graph_config3>(1,1*N_EDGES_MAX/4, 2*N_EDGES_MAX/4, effects_rep[1], influence_rep[1], receivers, senders, e_tmp[1], q_tmp[1], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  nnet::relational_model<input_t, index_t, input_t, graph_config3>(2,2*N_EDGES_MAX/4, 3*N_EDGES_MAX/4, effects_rep[2], influence_rep[2], receivers, senders, e_tmp[2], q_tmp[2], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  nnet::relational_model<input_t, index_t, input_t, graph_config3>(3,3*N_EDGES_MAX/4, 4*N_EDGES_MAX/4, effects_rep[3], influence_rep[3], receivers, senders, e_tmp[3], q_tmp[3], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
  
  nnet::merge_relation<input_t, result_t, graph_config3>(e_tmp, e);

}

