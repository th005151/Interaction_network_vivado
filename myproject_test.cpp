#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#include "parameters.h"
#include "myproject.h"
#include "nnet_helpers.h"


int main(int argc, char **argv)
{

  //hls-fpga-machine-learning insert data
  input_t      N_str[N_NODES_MAX][N_FEATURES];
  input_t      E_str[N_EDGES_MAX][E_FEATURES];
  index_t      receivers_str[N_EDGES_MAX][1];
  index_t      senders_str[N_EDGES_MAX][1];
  float        e_expected[N_EDGES_MAX][1];
#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);
  std::ifstream Nin("tb_input_node_features.dat");
  float num_n;
  std::vector<input_t> N_in;
  while (Nin >> num_n){
    N_in.push_back(num_n);
  }
  for(int i = 0; i < N_NODES_MAX; i++){
    for(int j = 0; j < N_FEATURES; j++){
      if(i < N_in.size()){
	N_str[i][j] = N_in[i*N_FEATURES + j];
      }else{
	N_str[i][j] = 0;
      }
      //std::cout << "N input = " << N_in[i*N_FEATURES + j] << std::endl;
    }
  }

  std::ifstream Ein("tb_input_edge_features.dat");
  float num_e;
  std::vector<input_t> E_in;
  while (Ein >> num_e){
    E_in.push_back(num_e);
  }
  for(int i = 0; i < N_EDGES_MAX; i++){
    for(int j = 0; j < E_FEATURES; j++){
      if(i < E_in.size()){
	E_str[i][j] = E_in[i*E_FEATURES + j];
      }else{
	E_str[i][j] = 0;
      }
      //std::cout << "E input = " << E_in[i*E_FEATURES + j] << std::endl;
    }
  }

  std::ifstream rin("tb_receivers.dat");
  int num_r;
  std::vector<index_t> r_in;
  while (rin >> num_r){
    r_in.push_back(num_r);
  }
  for(int i = 0; i < N_EDGES_MAX; i++){
    if(i < r_in.size()){
      receivers_str[i][0] = r_in[i];
    }else{
      receivers_str[i][0] = N_NODES_MAX - 1;
    }
    //std::cout << "r input = " << r_in[i] << std::endl;
  }

  std::ifstream sin("tb_senders.dat");
  int num_s;
  std::vector<index_t> s_in;
  while (sin >> num_s){
    s_in.push_back(num_s);
  }
  for(int i = 0; i < N_EDGES_MAX; i++){
    if(i < s_in.size()){
      senders_str[i][0] = s_in[i];
    }else{
      senders_str[i][0] = N_NODES_MAX - 1;
    }
    //std::cout << "s input = " << s_in[i] << std::endl;
  }

  std::ifstream ein("tb_output_edge_predictions.dat");
  float num_out;
  std::vector<float> e_in;
  while (ein >> num_out){
    e_in.push_back(num_out);
  }
  for(int i = 0; i < N_EDGES_MAX; i++){
    if(i < e_in.size()){
      e_expected[i][0] = e_in[i];
    }else{
      e_expected[i][0] = 0;
    }
    //std::cout << "e expected = " << e_in[i] << std::endl;
  }

  result_t e_str[N_EDGES_MAX][1];
  for(int i=0; i<N_EDGES_MAX; i++){
    e_str[i][0]=0;
  }
  
  unsigned short size_in, size_out;
  myproject(N_str, E_str, receivers_str, senders_str, e_str, size_in, size_out);
    
  std::cout << "e = " << std::endl;
  for(int i=0; i<N_EDGES_MAX; i++){
    //std::cout << e_str[i][0] << " ";
    if (e_str[i][0]==0 && e_expected[i][0]==0){
      std::cout << e_str[i][0] << " (expected " << e_expected[i][0] << ", 0 percent difference)" << std::endl;
      fout << e_str[i][0] << " (expected " << e_expected[i][0] << ", 0 percent difference)" << std::endl;
    }
    else {
      std::cout << e_str[i][0] << " (expected " << e_expected[i][0] << ", " << 100.*((float)e_str[i][0]-(float)e_expected[i][0])/(float)e_expected[i][0] << " percent difference)" << std::endl;
      fout << e_str[i][0] << " (expected " << e_expected[i][0] << ", " << 100.*((float)e_str[i][0]-(float)e_expected[i][0])/(float)e_expected[i][0] << " percent difference)" << std::endl;
    }
  }
  std::cout << std::endl;
  fout<<std::endl;

  std::ofstream eout;
  eout.open("tb_output_edge_labels.dat");
  if(e_in.size() < N_EDGES_MAX){
    for(int i = 0; i < e_in.size(); i++){
      eout << e_str[i][0] << " ";
    }
  }else{
    for(int i = 0; i < N_EDGES_MAX; i++){
      eout << e_str[i][0] << " ";
    }
  }
  //std::cout << "done writing to file" << std::endl;
  eout.close();
  fout.close();
  return 0;
}

