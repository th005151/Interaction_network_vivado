//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#ifndef NNET_GRAPH_H_
#define NNET_GRAPH_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_dense_large.h"
#include "nnet_activation.h"
#include "nnet_conv.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {
  
  struct graph_config
  {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    
    // Layer Sizes
    static const unsigned n_node = 10;
    static const unsigned n_edge = 20;
    static const unsigned n_batch = 20;
    static const unsigned n_in = 7;
    static const unsigned n_hidden = 4;
    static const unsigned n_out = 4;
    
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned io_stream = false;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
  };

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void IN_edge_module(
			data_T    Re[CONFIG_T::n_edge][CONFIG_T::n_hidden],
			data_T    Rn[CONFIG_T::n_node][CONFIG_T::n_hidden],
			index_T    receivers[CONFIG_T::n_edge][1],
			index_T    senders[CONFIG_T::n_edge][1],
			res_T     L[CONFIG_T::n_edge][CONFIG_T::n_hidden],
			res_T     Q[CONFIG_T::n_node][CONFIG_T::n_hidden],
			typename CONFIG_T::dense_config1::weight_t  core_edge_w0[3*CONFIG_T::n_hidden*CONFIG_T::n_hidden],
			typename CONFIG_T::dense_config1::bias_t    core_edge_b0[CONFIG_T::n_hidden],
			typename CONFIG_T::dense_config2::weight_t  core_edge_w1[CONFIG_T::n_hidden*CONFIG_T::n_hidden],
			typename CONFIG_T::dense_config2::bias_t    core_edge_b1[CONFIG_T::n_hidden])
  {
    if(CONFIG_T::io_stream){
      #pragma HLS STREAM variable=receivers
      #pragma HLS STREAM variable=senders
    }
    
    for(int i = 0; i < CONFIG_T::n_node; i++){
      for(int j = 0; j < CONFIG_T::n_hidden; j++){
	Q[i][j] = 0;
      }
    }

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor //*CONFIG_T::n_edge
    IN_edge_loop: for(int i = 0; i < CONFIG_T::n_edge; i++) {
      #pragma HLS UNROLL
      index_T r = receivers[i][0];
      index_T s = senders[i][0];
      data_T l_logits[2*CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=l_logits complete dim=0
      nnet::merge<data_T, CONFIG_T::n_hidden, CONFIG_T::n_hidden>(Re[i], Rn[r], l_logits);
      data_T l[3*CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=l complete dim=0
      nnet::merge<data_T, 2*CONFIG_T::n_hidden, CONFIG_T::n_hidden>(l_logits, Rn[s], l);

      data_T L0_logits[CONFIG_T::dense_config1::n_out];
      #pragma HLS ARRAY_PARTITION variable=L0_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config1>(l, L0_logits, core_edge_w0, core_edge_b0);
      data_T L0[CONFIG_T::relu_config1::n_in];
      #pragma HLS ARRAY_PARTITION variable=L0 complete dim=0
      nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(L0_logits, L0);

      data_T L_logits[CONFIG_T::dense_config2::n_out];
      #pragma HLS ARRAY_PARTITION variable=L_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config2>(L0, L_logits, core_edge_w1, core_edge_b1);
      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(L_logits, L[i]);

      for(int j = 0; j < CONFIG_T::n_hidden; j++){
        #pragma HLS UNROLL
	Q[r][j] += L[i][j];
      }
    }
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void IN_node_module(
			data_T    Rn[CONFIG_T::n_node][CONFIG_T::n_hidden],
			data_T    Q[CONFIG_T::n_node][CONFIG_T::n_hidden],
			res_T     P[CONFIG_T::n_node][CONFIG_T::n_hidden],
			typename CONFIG_T::dense_config1::weight_t  core_node_w0[2*CONFIG_T::n_hidden*CONFIG_T::n_hidden],
			typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::n_hidden],
			typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::n_hidden*CONFIG_T::n_hidden],
			typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::n_hidden])
  {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor //*CONFIG_T::n_node
    IN_node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){
      #pragma HLS UNROLL
      data_T p[2*CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=p complete dim=0
      nnet::merge<data_T, CONFIG_T::n_hidden, CONFIG_T::n_hidden>(Q[i], Rn[i], p);
      
      data_T P0_logits[CONFIG_T::dense_config1::n_out];
      #pragma HLS ARRAY_PARTITION variable=P0_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config1>(p, P0_logits, core_node_w0, core_node_b0);
      data_T P0[CONFIG_T::relu_config1::n_in];
      #pragma HLS ARRAY_PARTITION variable=P0 complete dim=0
      nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(P0_logits, P0);
      
      data_T P_logits[CONFIG_T::dense_config2::n_out];
      #pragma HLS ARRAY_PARTITION variable=P_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config2>(P0, P_logits, core_node_w1, core_node_b1);
      nnet::relu<data_T, res_T, typename CONFIG_T::relu_config1>(P_logits, P[i]);
    }
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void graph_independent(
			   data_T    X[CONFIG_T::dense_config1::n_batch][CONFIG_T::dense_config1::n_in],
			   res_T     R[CONFIG_T::dense_config2::n_batch][CONFIG_T::dense_config2::n_out],
			   typename CONFIG_T::dense_config1::weight_t  w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			   typename CONFIG_T::dense_config1::bias_t    b0[CONFIG_T::dense_config1::n_out],
			   typename CONFIG_T::dense_config2::weight_t  w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			   typename CONFIG_T::dense_config2::bias_t    b1[CONFIG_T::dense_config2::n_out])
  {
    if(CONFIG_T::io_stream){
      #pragma HLS STREAM variable=X
    }
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor //*CONFIG_T::dense_config2::n_batch
    data_T R0_logits[CONFIG_T::dense_config1::n_batch][CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=R0_logits complete dim=0
    nnet::dense_batch<data_T, data_T, typename CONFIG_T::dense_config1>(X, R0_logits, w0, b0);
    data_T R0[CONFIG_T::relu_config1::n_batch][CONFIG_T::relu_config1::n_in];
    #pragma HLS ARRAY_PARTITION variable=R0 complete dim=0
    nnet::relu_batch<data_T, data_T, typename CONFIG_T::relu_config1>(R0_logits, R0);

    if(CONFIG_T::activate_final){
        data_T R_logits[CONFIG_T::dense_config2::n_batch][CONFIG_T::dense_config2::n_out];
        #pragma HLS ARRAY_PARTITION variable=R_logits complete dim=0
        nnet::dense_batch<data_T, data_T, typename CONFIG_T::dense_config2>(R0, R_logits, w1, b1);
        nnet::relu_batch<data_T, res_T, typename CONFIG_T::relu_config2>(R_logits, R);
    }else{
      nnet::dense_batch<data_T, data_T, typename CONFIG_T::dense_config2>(R0, R, w1, b1);
    }
  }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void relational_model(
			  data_T    edge_terms[CONFIG_T::n_edge][CONFIG_T::e_features],
			  data_T    node_terms[CONFIG_T::n_node][CONFIG_T::n_features],
			  index_T   receivers[CONFIG_T::n_edge][1],
			  index_T   senders[CONFIG_T::n_edge][1],
			  res_T     effects[CONFIG_T::n_edge][CONFIG_T::n_out],
			  res_T     aggregation[CONFIG_T::n_node][CONFIG_T::n_out],
			  typename CONFIG_T::dense_config1::weight_t core_edge_w0[CONFIG_T::e_features*CONFIG_T::n_hidden + 2*CONFIG_T::n_features*CONFIG_T::n_hidden],
			  typename CONFIG_T::dense_config1::bias_t core_edge_b0[CONFIG_T::n_hidden],
			  typename CONFIG_T::dense_config2::weight_t core_edge_w1[CONFIG_T::n_hidden*CONFIG_T::n_hidden],
			  typename CONFIG_T::dense_config2::bias_t core_edge_b1[CONFIG_T::n_hidden],
			  typename CONFIG_T::dense_config2::weight_t core_edge_w2[CONFIG_T::n_hidden*CONFIG_T::n_hidden],
                          typename CONFIG_T::dense_config2::bias_t core_edge_b2[CONFIG_T::n_hidden],
			  typename CONFIG_T::dense_config3::weight_t core_edge_w3[CONFIG_T::n_hidden*CONFIG_T::n_out],
                          typename CONFIG_T::dense_config3::bias_t core_edge_b3[CONFIG_T::n_out])
  {
    // if(CONFIG_T::io_stream){
    //   #pragma HLS STREAM variable=receivers
    //   #pragma HLS STREAM variable=senders
    // }

	//#pragma HLS ARRAY_PARTITION variable=IsInit dim=1 complete
    // for(int i = 0; i < CONFIG_T::n_node; i++){
    //   for(int j = 0; j < CONFIG_T::n_out; j++){
	//       aggregation[i][j] = 0;
    //   }
    // }
    ap_uint<CONFIG_T::n_node> IsInit = 0;
	//bool IsInit[CONFIG_T::n_node] = {0};
  #pragma HLS DATAFLOW
	res_T aggregation_tmp[CONFIG_T::n_node][CONFIG_T::n_out];
	#pragma HLS ARRAY_PARTITION variable=aggregation_tmp complete dim=0
	
    IN_relational_loop: for(int i = 0; i < CONFIG_T::n_edge; i++){
      #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
      index_T r = receivers[i][0];
      index_T s = senders[i][0];
      data_T reference_logits[CONFIG_T::e_features + CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=reference_logits complete dim=0
      nnet::merge<data_T, CONFIG_T::e_features, CONFIG_T::n_features>(edge_terms[i], node_terms[r], reference_logits);

      data_T reference[CONFIG_T::e_features + 2*CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=reference complete dim=0
      nnet::merge<data_T, CONFIG_T::e_features + CONFIG_T::n_features, CONFIG_T::n_features>(reference_logits, node_terms[s], reference);

      data_T effects0_logits[CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=effects0_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config1>(reference, effects0_logits, core_edge_w0, core_edge_b0);
      data_T effects0[CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=effects0 complete dim=0
      nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(effects0_logits, effects0);

      data_T effects1_logits[CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=effects1_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config2>(effects0, effects1_logits, core_edge_w1, core_edge_b1);
      data_T effects1[CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=effects1 complete dim=0
      nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(effects1_logits, effects1);

      data_T effects2_logits[CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=effects2_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config2>(effects1, effects2_logits, core_edge_w2, core_edge_b2);
      data_T effects2[CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=effects2 complete dim=0
      nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(effects2_logits, effects2);

      if(CONFIG_T::activate_final){
        data_T effects_logits[CONFIG_T::n_out];
        #pragma HLS ARRAY_PARTITION variable=effects_logits complete dim=0
	nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config3>(effects2, effects_logits, core_edge_w3, core_edge_b3);
	nnet::sigmoid<data_T, res_T, typename CONFIG_T::sigmoid_config1>(effects_logits, effects[i]);
      }else{
	nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config3>(effects2, effects[i], core_edge_w3, core_edge_b3);
      }

	// for(int j = 0; j < CONFIG_T::n_out; j++){
	// 	if(IsInit[r]){
	// 		aggregation[r][j] += effects[i][j];
	// 	}
	// 	else{
	// 		aggregation[r][j] = effects[i][j];
	// 		IsInit[r] = 1;
	// 	}
	// }

		for(int j = 0; j < CONFIG_T::n_out; j++){
			if(IsInit[r]){
				aggregation_tmp[r][j] += effects[i][j];
			}
			else{
				aggregation_tmp[r][j] = effects[i][j];
				IsInit[r] = 1;
			}
		}


    }

	for(int i = 0; i<CONFIG_T::n_node; i++){
		for(int j = 0; j < CONFIG_T::n_out; j++){
      #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
			aggregation[i][j] = aggregation_tmp[i][j];
		}
	}

  }

  template<class data_T, class res_T, typename CONFIG_T>
    void object_model(
		      data_T    node_terms[CONFIG_T::n_node][CONFIG_T::n_features],
                      data_T    aggregation[CONFIG_T::n_node][CONFIG_T::n_in],
	              res_T     influence[CONFIG_T::n_node][CONFIG_T::n_out],
		      typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::n_in*CONFIG_T::n_hidden + CONFIG_T::n_features*CONFIG_T::n_hidden],
		      typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::n_hidden],
		      typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::n_hidden*CONFIG_T::n_hidden],
		      typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::n_hidden],
		      typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::n_hidden*CONFIG_T::n_out],
		      typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::n_out])
  {
    
    IN_object_loop: for(int i = 0; i < CONFIG_T::n_node; i++){
      #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
      data_T aggregated[CONFIG_T::n_in + CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=aggregated complete dim=0
      nnet::merge<data_T, CONFIG_T::n_in, CONFIG_T::n_features>(aggregation[i], node_terms[i], aggregated);

      data_T influence0_logits[CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=influence0_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config1>(aggregated, influence0_logits, core_node_w0, core_node_b0);
      data_T influence0[CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=influence0 complete dim=0
      nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(influence0_logits, influence0);

      data_T influence1_logits[CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=influence1_logits complete dim=0
      nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config2>(influence0, influence1_logits, core_node_w1, core_node_b1);
      data_T influence1[CONFIG_T::n_hidden];
      #pragma HLS ARRAY_PARTITION variable=influence1 complete dim=0
      nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(influence1_logits, influence1);

      if(CONFIG_T::activate_final){
        data_T influence_logits[CONFIG_T::n_out];
        #pragma HLS ARRAY_PARTITION variable=influence_logits complete dim=0
        nnet::dense_large_basic<data_T, data_T, typename CONFIG_T::dense_config3>(influence1, influence_logits, core_node_w2, core_node_b2);
        nnet::relu<data_T, res_T, typename CONFIG_T::relu_config2>(influence_logits, influence[i]);
      }else{
        nnet::dense_large_basic<data_T, res_T, typename CONFIG_T::dense_config3>(influence1, influence[i], core_node_w2, core_node_b2);
      }
    }
  }
	/*
  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void interaction_network(
                             data_T    E[CONFIG_T::n_edge][CONFIG_T::e_features],
                             data_T    N[CONFIG_T::n_node][CONFIG_T::n_features],
                             index_T    receivers[CONFIG_T::n_edge][1],
                             index_T    senders[CONFIG_T::n_edge][1],
			     res_T     predicted[CONFIG_T::n_edge][1],
			     typename CONFIG_T::graph_config2::dense_config1::weight_t  core_edge_w0[CONFIG_T::graph_config2::n_in*CONFIG_T::graph_config2::n_hidden],
			     typename CONFIG_T::graph_config2::dense_config1::bias_t    core_edge_b0[CONFIG_T::graph_config2::n_hidden],
			     typename CONFIG_T::graph_config2::dense_config2::weight_t  core_edge_w1[CONFIG_T::graph_config2::n_hidden*CONFIG_T::graph_config2::n_hidden],
			     typename CONFIG_T::graph_config2::dense_config2::bias_t    core_edge_b1[CONFIG_T::graph_config2::n_hidden],
			     typename CONFIG_T::graph_config2::dense_config2::weight_t  core_edge_w2[CONFIG_T::graph_config2::n_hidden*CONFIG_T::graph_config2::n_hidden],
			     typename CONFIG_T::graph_config2::dense_config2::bias_t    core_edge_b2[CONFIG_T::graph_config2::n_hidden],
			     typename CONFIG_T::graph_config2::dense_config3::weight_t  core_edge_w3[CONFIG_T::graph_config2::n_hidden*CONFIG_T::graph_config2::n_out],
			     typename CONFIG_T::graph_config2::dense_config3::bias_t    core_edge_b3[CONFIG_T::graph_config2::n_out],
			     typename CONFIG_T::graph_config3::dense_config1::weight_t  core_node_w0[CONFIG_T::graph_config3::n_in*CONFIG_T::graph_config3::n_hidden],
			     typename CONFIG_T::graph_config3::dense_config1::bias_t    core_node_b0[CONFIG_T::graph_config3::n_hidden],
			     typename CONFIG_T::graph_config3::dense_config2::weight_t  core_node_w1[CONFIG_T::graph_config3::n_hidden*CONFIG_T::graph_config3::n_hidden],
			     typename CONFIG_T::graph_config3::dense_config2::bias_t    core_node_b1[CONFIG_T::graph_config3::n_hidden],
			     typename CONFIG_T::graph_config3::dense_config3::weight_t  core_node_w2[CONFIG_T::graph_config3::n_hidden*CONFIG_T::graph_config3::n_out],
			     typename CONFIG_T::graph_config3::dense_config3::bias_t    core_node_b2[CONFIG_T::graph_config3::n_out],
			     typename CONFIG_T::graph_config4::dense_config1::weight_t  core_final_w0[CONFIG_T::graph_config4::n_in*CONFIG_T::graph_config4::n_hidden],
                             typename CONFIG_T::graph_config4::dense_config1::bias_t    core_final_b0[CONFIG_T::graph_config4::n_hidden],
                             typename CONFIG_T::graph_config4::dense_config2::weight_t  core_final_w1[CONFIG_T::graph_config4::n_hidden*CONFIG_T::graph_config4::n_hidden],
                             typename CONFIG_T::graph_config4::dense_config2::bias_t    core_final_b1[CONFIG_T::graph_config4::n_hidden],
                             typename CONFIG_T::graph_config4::dense_config2::weight_t  core_final_w2[CONFIG_T::graph_config4::n_hidden*CONFIG_T::graph_config4::n_hidden],
                             typename CONFIG_T::graph_config4::dense_config2::bias_t    core_final_b2[CONFIG_T::graph_config4::n_hidden],
                             typename CONFIG_T::graph_config4::dense_config3::weight_t  core_final_w3[CONFIG_T::graph_config4::n_hidden*CONFIG_T::graph_config4::n_out],
                             typename CONFIG_T::graph_config4::dense_config3::bias_t    core_final_b3[CONFIG_T::graph_config4::n_out])
  {
    if(CONFIG_T::io_stream){
      #pragma HLS STREAM variable=E
      #pragma HLS STREAM variable=N
      #pragma HLS STREAM variable=receivers
      #pragma HLS STREAM variable=senders
    }

    data_T interaction_terms[CONFIG_T::n_edge][CONFIG_T::e_features+2*CONFIG_T::n_features];
    #pragma HLS ARRAY_PARTITION variable=interaction_terms complete dim=0

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    Edge_reference_loop: for(int i = 0; i < CONFIG_T::n_edge; i++){
      #pragma HLS UNROLL
      index_T r = receivers[i][0];
      index_T s = senders[i][0];
      data_T l_logits[CONFIG_T::e_features+CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=l_logits complete dim=0
      nnet::merge<data_T, CONFIG_T::e_features, CONFIG_T::n_features>(E[i], N[r], l_logits);
      nnet::merge<data_T, CONFIG_T::e_features+CONFIG_T::n_features, CONFIG_T::n_features>(l_logits, N[s], interaction_terms[i]);
    }

    data_T effects[CONFIG_T::n_edge][CONFIG_T::n_hidden];
    #pragma HLS ARRAY_PARTITION variable=effects complete dim=0
    nnet::relational_model<data_T, res_T, typename CONFIG_T::graph_config2>(interaction_terms, effects, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);

    data_T aggregation[CONFIG_T::n_node][CONFIG_T::n_hidden];
    #pragma HLS ARRAY_PARTITION variable=aggregation complete dim=0
    for(int i = 0; i < CONFIG_T::n_node; i++){
      for(int j = 0; j < CONFIG_T::n_hidden; j++){
        aggregation[i][j] = 0;
      }
    }
    for(int i = 0; i < CONFIG_T::n_edge; i++){
      for(int j = 0; j < CONFIG_T::n_hidden; j++){
	#pragma HLS UNROLL
        index_T r = receivers[i][0];
        aggregation[r][j] += effects[i][j];
      }
    }

    data_T aggregated[CONFIG_T::n_node][CONFIG_T::n_hidden+CONFIG_T::n_features];
    #pragma HLS ARRAY_PARTITION variable=aggregated complete dim=0

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    Node_aggregation_loop: for(int i = 0; i < CONFIG_T::n_node; i++){
      #pragma HLS UNROLL
      nnet::merge<data_T, CONFIG_T::n_hidden, CONFIG_T::n_features>(aggregation[i], N[i], aggregated[i]);
    }

    data_T influence[CONFIG_T::n_node][CONFIG_T::n_features];
    #pragma HLS ARRAY_PARTITION variable=influence complete dim=0
    nnet::object_model<data_T, res_T, typename CONFIG_T::graph_config3>(aggregated, influence, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    Classification_loop: for(int i = 0; i < CONFIG_T::n_edge; i++){
      #pragma HLS UNROLL
      index_T r = receivers[i][0];
      index_T s = senders[i][0];
      data_T p_logits[CONFIG_T::e_features+CONFIG_T::n_features];
      #pragma HLS ARRAY_PARTITION variable=p_logits complete dim=0
      nnet::merge<data_T, CONFIG_T::e_features, CONFIG_T::n_features>(E[i], influence[r], p_logits);
      nnet::merge<data_T, CONFIG_T::e_features+CONFIG_T::n_features, CONFIG_T::n_features>(p_logits, influence[s], interaction_terms[i]);
    }

    #pragma HLS ARRAY_PARTITION variable=predicted complete dim=0
    nnet::relational_model<data_T, res_T, typename CONFIG_T::graph_config4>(interaction_terms, predicted, core_final_w0, core_final_b0, core_final_w1, core_final_b1, core_final_w2, core_final_b2, core_final_w3, core_final_b3);
  }
	*/

  template<class data_T, class res_T, typename CONFIG_T>
    void compute_edge_net_features(
		   data_T    X[CONFIG_T::n_node][CONFIG_T::n_input_dim],
		   ap_uint<1> Ri[CONFIG_T::n_node][CONFIG_T::n_edge],
		   ap_uint<1> Ro[CONFIG_T::n_node][CONFIG_T::n_edge],
		   res_T     B[CONFIG_T::n_edge][2*CONFIG_T::n_input_dim])
  {
    data_T bo[CONFIG_T::n_edge][CONFIG_T::n_input_dim];
    data_T bi[CONFIG_T::n_edge][CONFIG_T::n_input_dim];
    
    if (CONFIG_T::io_type == io_parallel){
      // For parallel inputs:
      //   - completely partition arrays -- target fabric
      //   - if we have an unroll factor, limit number of multipliers
    	#pragma HLS PIPELINE II=CONFIG_T::reuse_factor
		#pragma HLS ARRAY_PARTITION variable=bo complete
		#pragma HLS ARRAY_PARTITION variable=bi complete
    }
    else if (CONFIG_T::io_type == io_serial){
		#pragma HLS DATAFLOW
		#pragma HLS STREAM variable=bo depth=1
		#pragma HLS STREAM variable=bi depth=1
    }

    // Multiply bo = Ro^T X
    // Multiply bi = Ri^T X
    for(int ii = 0; ii < CONFIG_T::n_edge; ii++) {
      if (CONFIG_T::io_type == io_serial){
    	  #pragma HLS PIPELINE
      }
      for(int jj = 0; jj < CONFIG_T::n_input_dim; jj++) {
    	  bi[ii][jj] = 0;
    	  bo[ii][jj] = 0;
    	  for(int kk = 0; kk < CONFIG_T::n_node; kk++) {
    		  bi[ii][jj] += Ri[kk][ii] * X[kk][jj];
    		  bo[ii][jj] += Ro[kk][ii] * X[kk][jj];
    	  }
    	  // Concatenate M = [bo, bi]
    	  B[ii][jj] = (res_T) bo[ii][jj];
    	  B[ii][CONFIG_T::n_input_dim+jj] = (res_T) bi[ii][jj];
      }
    }
  }

  template<class data_T, class res_T, typename CONFIG_T>
  void compute_node_net_features(
		  data_T    X[CONFIG_T::n_node][CONFIG_T::n_input_dim],
		  data_T    e[CONFIG_T::n_edge][1],
		  ap_uint<1> Ri[CONFIG_T::n_node][CONFIG_T::n_edge],
		  ap_uint<1> Ro[CONFIG_T::n_node][CONFIG_T::n_edge],
		  res_T     M[CONFIG_T::n_node][3*CONFIG_T::n_input_dim])
  {
	  data_T bo[CONFIG_T::n_edge][CONFIG_T::n_input_dim];
	  data_T bi[CONFIG_T::n_edge][CONFIG_T::n_input_dim];
	  data_T mo[CONFIG_T::n_node][CONFIG_T::n_input_dim];
	  data_T mi[CONFIG_T::n_node][CONFIG_T::n_input_dim];
	  data_T Rwo[CONFIG_T::n_node][CONFIG_T::n_edge];
	  data_T Rwi[CONFIG_T::n_node][CONFIG_T::n_edge];

	  if (CONFIG_T::io_type == io_parallel){
      // For parallel inputs:
      //   - completely partition arrays -- target fabric
      //   - if we have an unroll factor, limit number of multipliers
		  #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
		  #pragma HLS ARRAY_PARTITION variable=bo complete
		  #pragma HLS ARRAY_PARTITION variable=bi complete
		  #pragma HLS ARRAY_PARTITION variable=mo complete
		  #pragma HLS ARRAY_PARTITION variable=mi complete
		  #pragma HLS ARRAY_PARTITION variable=Rwo complete
		  #pragma HLS ARRAY_PARTITION variable=Rwi complete

		  int multiplier_limit  = ceil(float(2*CONFIG_T::n_node*CONFIG_T::n_edge*CONFIG_T::n_input_dim) / float(CONFIG_T::reuse_factor));
		  #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
	  }
	  else if (CONFIG_T::io_type == io_serial)
	  {
		  #pragma HLS DATAFLOW
		  #pragma HLS STREAM variable=bo depth=1
		  #pragma HLS STREAM variable=bi depth=1
		  #pragma HLS STREAM variable=mo depth=1
		  #pragma HLS STREAM variable=mi depth=1
		  #pragma HLS STREAM variable=Rwo depth=1
		  #pragma HLS STREAM variable=Rwi depth=1
	  }

	  // Multiply bo = Ro^T X
	  // Multiply bi = Ri^T X
	  for(int ii = 0; ii < CONFIG_T::n_edge; ii++)
	  {
		  if (CONFIG_T::io_type == io_serial)
		  {
			  #pragma HLS PIPELINE
		  }
		  for(int jj = 0; jj < CONFIG_T::n_input_dim; jj++)
		  {
			  bi[ii][jj] = 0;
			  bo[ii][jj] = 0;
			  for(int kk = 0; kk < CONFIG_T::n_node; kk++)
			  {
				  bi[ii][jj] += Ri[kk][ii] * X[kk][jj];
				  bo[ii][jj] += Ro[kk][ii] * X[kk][jj];
			  }
		  }
	  }

	  // Hadamard Multiply Rwo = Ro * e
	  // Hadamard Multiply Rwi = Ri * e
	  for(int ii = 0; ii < CONFIG_T::n_node; ii++)
	  {
		  if (CONFIG_T::io_type == io_serial)
		  {
			  #pragma HLS PIPELINE
			  }
		  for(int jj = 0; jj < CONFIG_T::n_edge; jj++)
		  {
			  Rwo[ii][jj] = Ro[ii][jj] * e[jj][0];
			  Rwi[ii][jj] = Ri[ii][jj] * e[jj][0];
		  }
	  }

	  // Multiply mi = Rwi bo
	  // Multiply mo = Rwo bi
	  int counter = 0;
	  for(int ii = 0; ii < CONFIG_T::n_node; ii++)
	  {
		  if (CONFIG_T::io_type == io_serial)
		  {
			  #pragma HLS PIPELINE
			  }
		  for(int jj = 0; jj < CONFIG_T::n_input_dim; jj++)
		  {
			  mi[ii][jj] = 0;
			  mo[ii][jj] = 0;
			  for(int kk = 0; kk < CONFIG_T::n_edge; kk++)
			  {
				  counter+=1;
				  mi[ii][jj] += Rwi[ii][kk] * bo[kk][jj];
				  mo[ii][jj] += Rwo[ii][kk] * bi[kk][jj];
			  }
			  // Concatenate M = [mi, mo, X]
			  M[ii][jj] = (res_T) mi[ii][jj];
			  M[ii][CONFIG_T::n_input_dim+jj] = (res_T) mo[ii][jj];
			  M[ii][2*CONFIG_T::n_input_dim+jj] = (res_T) X[ii][jj];
		  }
	  }
  }
}

#endif
