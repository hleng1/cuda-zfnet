#include <stdio.h>
#include <stdlib.h>

#include "zf_kernel.cu"

const int INPUT_SIZE = 224 * 224 * 3;
const int LAYER_1_FILTER_SIZE = 7 * 7 * 3;
const int LAYER_1_FILTER_NUM = 96;
const int LAYER_2_INPUT_SIZE = 110 * 110 * 96;
const int LAYER_2_FILTER_SIZE = 5 * 5 * 96;
const int LAYER_2_FILTER_NUM = 256;
const int LAYER_2_POOLED_SIZE = 55 * 55 * 96;

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
  float *d_layer_1_input;
  float *d_layer_2_input;
  float *d_layer_1_weights;
  float *d_layer_2_pooled;

  cudaMalloc((void **)&d_layer_1_input, INPUT_SIZE * sizeof(float));
  cudaMalloc((void **)&d_layer_2_input, LAYER_2_INPUT_SIZE * sizeof(float));

  cudaMalloc((void **)&d_layer_1_weights, LAYER_1_FILTER_SIZE * LAYER_1_FILTER_NUM * sizeof(float));
  cudaMemcpy(d_layer_1_weights, layer_1_weights, LAYER_1_FILTER_SIZE * LAYER_1_FILTER_NUM * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_layer_2_pooled, LAYER_2_POOLED_SIZE * sizeof(float));


  // layer 1: 110 * 110 * 96
  dim3 conv_1_grid_dim(96, 1, 1);
  dim3 conv_1_block_dim(110, 110);

  printf("Running conv_1 ...\n");
  run_conv_1<<<conv_1_grid_dim, conv_1_block_dim>>>(d_layer_1_input, d_layer_1_weights, d_layer_2_input);

  dim3 pool_1_grid_dim(96, 1, 1);
  dim3 pool_1_block_dim(55, 55);
  printf("Running pool_1 ...\n");
  run_pool_1<<<pool_1_grid_dim, pool_1_block_dim>>>(d_layer_2_input, d_layer_2_pooled);

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