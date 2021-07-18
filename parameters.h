#ifndef PARAMETERS_H_
#define PARAMETERS_H_
#define AP_INT_MAX_W 10240
#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_dense.h"
#include "nnet_conv.h"
#include "nnet_activation.h"
#include "nnet_graph.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<16,6> accum_default_t;
typedef ap_fixed<16,6> weight_default_t;
typedef ap_fixed<16,6> bias_default_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> result_t;
typedef ap_uint<16> index_t;
#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)
#define REUSE_GRAPH 1 
#define REUSE_DENSE 1
#define N_ITERS 1
#define LATENT_EDGE 8
#define LATENT_NODE 8
#define N_FEATURES 3
#define E_FEATURES 1 
//graph_nets simple example:
#define N_NODES_MAX 28 //28 112
#define N_EDGES_MAX 40 //40 148 

//hls-fpga-machine-learning insert layer-config

struct graph_config1 : nnet::graph_config {
  static const unsigned n_edge = N_EDGES_MAX;
  static const unsigned n_node = N_NODES_MAX;
  static const unsigned n_hidden = LATENT_EDGE;
  static const unsigned e_features = E_FEATURES;
  static const unsigned n_features = N_FEATURES;
  static const unsigned n_out = E_FEATURES;
  static const bool io_stream = true;
  static const bool activate_final = true;
  static const unsigned reuse_factor = REUSE_GRAPH;

  struct dense_config1 : nnet::dense_config {
    static const unsigned n_in = e_features + 2*n_features;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_in = n_hidden;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct dense_config3 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = e_features; //n_out
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct sigmoid_config1 : nnet::activ_config {
    static const unsigned n_in = n_out;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
};

struct graph_config2 : nnet::graph_config {
  static const unsigned n_edge = N_EDGES_MAX;
  static const unsigned n_node = N_NODES_MAX;
  static const unsigned n_hidden = LATENT_NODE;
  static const unsigned e_features = E_FEATURES;
  static const unsigned n_features = N_FEATURES;
  static const unsigned n_in = E_FEATURES;
  static const unsigned n_out = N_FEATURES;
  static const bool activate_final = false;
  static const unsigned reuse_factor = REUSE_GRAPH;

  struct dense_config1 : nnet::dense_config {
    static const unsigned n_in = e_features + n_features; //n_in + n_features
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_in = n_hidden;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct dense_config3 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = n_features; //n_out
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config2 : nnet::activ_config {
    static const unsigned n_in = n_out;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
};

struct graph_config3 : nnet::graph_config {
  static const unsigned n_edge = N_EDGES_MAX;
  static const unsigned n_node = N_NODES_MAX;
  static const unsigned n_hidden = LATENT_EDGE;
  static const unsigned e_features = E_FEATURES;
  static const unsigned n_features = N_FEATURES;
  static const unsigned n_out = E_FEATURES;
  static const bool io_stream = true;
  static const bool activate_final = true;
  static const unsigned reuse_factor = REUSE_GRAPH;

  struct dense_config1 : nnet::dense_config {
    static const unsigned n_in = e_features + 2*n_features;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct relu_config1 : nnet::activ_config {
    static const unsigned n_in = n_hidden;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
  struct dense_config2 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = n_hidden;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct dense_config3 : nnet::dense_config {
    static const unsigned n_in = n_hidden;
    static const unsigned n_out = e_features; //n_out
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = REUSE_DENSE;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
  };
  struct sigmoid_config1 : nnet::activ_config {
    static const unsigned n_in = n_out;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
  };
};

#endif 
