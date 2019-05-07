#include <stdio.h>
#include <stdlib.h>

#include "zf_kernel.cu"

const int INPUT_SIZE = 224 * 224 * 3;

const int LAYER_1_INPUT_SIZE = 110 * 110 * 96;
const int LAYER_1_FILTER_SIZE = 7 * 7 * 3;
const int LAYER_1_FILTER_NUM = 96;
const int LAYER_1_POOLED_SIZE = 55 * 55 * 96;
const int LAYER_1_PADDED_SIZE = 57 * 57 * 96;

const int LAYER_2_INPUT_SIZE = 26 * 26 * 256;
const int LAYER_2_FILTER_SIZE = 5 * 5 * 96;
const int LAYER_2_FILTER_NUM = 256;
const int LAYER_2_POOLED_SIZE = 13 * 13 * 256;
const int LAYER_2_PADDED_SIZE = 15 * 15 * 256;

const int LAYER_3_INPUT_SIZE = 13 * 13 * 384;
const int LAYER_3_FILTER_SIZE = 3 * 3 * 256;
const int LAYER_3_FILTER_NUM = 384;

const int LAYER_4_INPUT_SIZE = 13 * 13 * 384;
const int LAYER_4_FILTER_SIZE = 3 * 3 * 384;
const int LAYER_4_FILTER_NUM = 384;

const int LAYER_5_INPUT_SIZE = 13 * 13 * 256;
const int LAYER_5_POOLED_SIZE = 6 * 6 * 256;

void read_file(const char *file_path, float *dest_array);

int main(int argc, char **argv) {

  // host
  float *input_array; // layer_1_input
  float *layer_1_weights;
  float *layer_2_weights;

  input_array = (float *)malloc(INPUT_SIZE * sizeof(float));
  layer_1_weights = (float *)malloc(LAYER_1_FILTER_SIZE * LAYER_1_FILTER_NUM * sizeof(float));
  layer_2_weights = (float *)malloc(LAYER_2_FILTER_SIZE * LAYER_2_FILTER_NUM * sizeof(float));

  // read_file("data/input.txt", input_array);
  read_file("data/layer1.txt", layer_1_weights);
  read_file("data/layer2.txt", layer_2_weights);



  // device 
  float *d_input;
  float *d_layer_1_input;
  float *d_layer_1_weights;
  float *d_layer_1_pooled;
  float *d_layer_1_padded;

  float *d_layer_2_input;
  float *d_layer_2_weights;
  float *d_layer_2_pooled;
  float *d_layer_2_padded;

  // input 
  cudaMalloc((void **)&d_input, INPUT_SIZE * sizeof(float));
  // layer 1
  cudaMalloc((void **)&d_layer_1_input, LAYER_1_INPUT_SIZE * sizeof(float));
  cudaMalloc((void **)&d_layer_1_weights, LAYER_1_FILTER_SIZE * LAYER_1_FILTER_NUM * sizeof(float));
  cudaMemcpy(d_layer_1_weights, layer_1_weights, LAYER_1_FILTER_SIZE * LAYER_1_FILTER_NUM * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_layer_1_pooled, LAYER_1_POOLED_SIZE * sizeof(float));
  
  cudaMalloc((void **)&d_layer_1_padded, LAYER_1_PADDED_SIZE * sizeof(float));
  cudaMemset(d_layer_1_padded, 0, LAYER_1_POOLED_SIZE * sizeof(float));

  // layer 2
  cudaMalloc((void **)&d_layer_2_input, LAYER_2_INPUT_SIZE * sizeof(float));

  cudaMalloc((void **)&d_layer_2_weights, LAYER_2_FILTER_SIZE * LAYER_2_FILTER_NUM * sizeof(float));
  cudaMemcpy(d_layer_2_weights, layer_2_weights, LAYER_2_FILTER_SIZE * LAYER_2_FILTER_NUM * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_layer_2_pooled, LAYER_2_POOLED_SIZE * sizeof(float));
  
  cudaMalloc((void **)&d_layer_2_padded, LAYER_2_PADDED_SIZE * sizeof(float));
  cudaMemset(d_layer_2_padded, 0, LAYER_2_POOLED_SIZE * sizeof(float));



  // layer 1: 110 * 110 * 96
  dim3 conv_1_grid_dim(96, 1, 1);
  dim3 conv_1_block_dim(110, 110);

  printf("Running conv_1 ...\n");
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv_1<<<conv_1_grid_dim, conv_1_block_dim>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110);

  dim3 pool_1_grid_dim(96, 1, 1);
  dim3 pool_1_block_dim(55, 55);
  printf("Running pool_1 ...\n");
  run_pool_1<<<pool_1_grid_dim, pool_1_block_dim>>>(d_layer_1_input, d_layer_1_pooled);

  dim3 pad_1_grid_dim(96, 1, 1);
  dim3 pad_1_block_dim(55, 55);
  printf("Padding pool_1 output ...\n");
  run_padding_1<<<pad_1_grid_dim, pad_1_block_dim>>>(d_layer_1_pooled, d_layer_1_padded, 55);

  dim3 lcn_1_grid_dim(96, 1, 1);
  dim3 lcn_1_block_dim(55, 55);
  printf("Running lcn_1\n");
  run_lcn_1<<<lcn_1_grid_dim, lcn_1_block_dim>>>(d_layer_1_padded, d_layer_1_pooled);

  // layer 2: 26 * 26 * 256
  dim3 conv_2_grid_dim(256, 1, 1);
  dim3 conv_2_block_dim(26, 26);

  printf("Running conv_2 ...\n");
  run_conv_2<<<conv_2_grid_dim, conv_2_block_dim>>>(d_input, d_layer_2_weights, d_layer_2_input);

  dim3 pool_2_grid_dim(256, 1, 1);
  dim3 pool_2_block_dim(26, 26);
  printf("Running pool_2 ...\n");
  run_pool_2<<<pool_2_grid_dim, pool_2_block_dim>>>(d_layer_2_input, d_layer_2_pooled);

  dim3 pad_2_grid_dim(256, 1, 1);
  dim3 pad_2_block_dim(26, 26);
  printf("Padding pool_2 output ...\n");
  run_padding_2<<<pad_2_grid_dim, pad_2_block_dim>>>(d_layer_2_pooled, d_layer_2_padded, 26);

  dim3 lcn_2_grid_dim(256, 1, 1);
  dim3 lcn_2_block_dim(26, 26);
  printf("Running lcn_2\n");
  run_lcn_2<<<lcn_2_grid_dim, lcn_2_block_dim>>>(d_layer_2_padded, d_layer_2_pooled);


}


void read_file(const char *file_path, float *dest_array) {
  FILE *fp = fopen(file_path, "r");
  int count = 0;
  char *line = NULL;
  size_t len = 0;
  ssize_t nread;
  if (fp == NULL) {
    perror("fopen");
    exit(EXIT_FAILURE);
  }
  while ((nread = getline(&line, &len, fp)) != -1) {
    printf("Read %zu\n", len);
    dest_array[count++] = atof(line);
  }
  free(line);
  fclose(fp);
}