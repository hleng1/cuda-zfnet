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
const int LAYER_5_FILTER_SIZE = 1 * 1 * 4096;

const int LAYER_6_INPUT_SIZE = 1 * 1 * 4096;
const int LAYER_6_FILTER_SIZE = 1 * 1 * 4096;

const int LAYER_7_INPUT_SIZE = 1 * 1 * 4096;
const int LAYER_7_FILTER_SIZE = 1 * 1 * 1000;

const int LAYER_8_INPUT_SIZE = 1 * 1 * 4096;

void read_file(const char *file_path, float *dest_array);

int main(int argc, char **argv) {

  // host
  float *input_array; // layer_1_input
  float *layer_1_weights;
  float *layer_2_weights;
  float *layer_3_weights;
  float *layer_4_weights;
  float *layer_5_weights;
  float *layer_6_weights;
  float *layer_7_weights;
  float *layer_8_weights;

  input_array = (float *)malloc(INPUT_SIZE * sizeof(float));

  layer_1_weights = (float *)malloc(LAYER_1_FILTER_SIZE * LAYER_1_FILTER_NUM * sizeof(float));
  layer_2_weights = (float *)malloc(LAYER_2_FILTER_SIZE * LAYER_2_FILTER_NUM * sizeof(float));
  layer_3_weights = (float *)malloc(LAYER_3_FILTER_SIZE * LAYER_3_FILTER_NUM * sizeof(float));
  layer_4_weights = (float *)malloc(LAYER_4_FILTER_SIZE * LAYER_4_FILTER_NUM * sizeof(float));
  layer_5_weights = (float *)malloc(LAYER_5_FILTER_SIZE * LAYER_5_FILTER_NUM * sizeof(float));
  layer_6_weights = (float *)malloc(LAYER_6_FILTER_SIZE * LAYER_6_FILTER_NUM * sizeof(float));
  layer_7_weights = (float *)malloc(LAYER_7_FILTER_SIZE * LAYER_7_FILTER_NUM * sizeof(float));
  layer_8_weights = (float *)malloc(LAYER_8_FILTER_SIZE * LAYER_8_FILTER_NUM * sizeof(float));

  read_file("data/input.txt", input_array);
  read_file("data/layer1.txt", layer_1_weights);
  read_file("data/layer2.txt", layer_2_weights);
  read_file("data/layer3.txt", layer_3_weights);
  read_file("data/layer4.txt", layer_4_weights);
  read_file("data/layer5.txt", layer_5_weights);
  read_file("data/layer6.txt", layer_6_weights);
  read_file("data/layer7.txt", layer_7_weights);
  read_file("data/layer8.txt", layer_8_weights);



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

  float *d_layer_3_input;
  float *d_layer_3_weights;

  float *d_layer_4_input;
  float *d_layer_4_weights;

  float *d_layer_5_input;
  float *d_layer_5_pooled;
  float *d_layer_5_weights;

  float *d_layer_6_input;
  float *d_layer_6_weights;

  float *d_layer_7_input;
  float *d_layer_7_weights;

  float *d_layer_8_input;

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


  // layer 3
  cudaMalloc((void **)&d_layer_3_input, LAYER_3_INPUT_SIZE * sizeof(float));

  cudaMalloc((void **)&d_layer_3_weights, LAYER_3_FILTER_SIZE * LAYER_3_FILTER_NUM * sizeof(float));
  cudaMemcpy(d_layer_3_weights, layer_3_weights, LAYER_3_FILTER_SIZE * LAYER_3_FILTER_NUM * sizeof(float), cudaMemcpyHostToDevice);

  // layer 4
  cudaMalloc((void **)&d_layer_4_input, LAYER_4_INPUT_SIZE * sizeof(float));

  cudaMalloc((void **)&d_layer_4_weights, LAYER_4_FILTER_SIZE * LAYER_4_FILTER_NUM * sizeof(float));
  cudaMemcpy(d_layer_4_weights, layer_4_weights, LAYER_4_FILTER_SIZE * LAYER_4_FILTER_NUM * sizeof(float), cudaMemcpyHostToDevice);


  // layer 5
  cudaMalloc((void **)&d_layer_5_input, LAYER_5_INPUT_SIZE * sizeof(float));

  cudaMalloc((void **)&d_layer_5_weights, LAYER_5_FILTER_SIZE * LAYER_5_FILTER_NUM * sizeof(float));
  cudaMemcpy(d_layer_5_weights, layer_5_weights, LAYER_5_FILTER_SIZE * LAYER_5_FILTER_NUM * sizeof(float), cudaMemcpyHostToDevice);


  // layer 6
  cudaMalloc((void **)&d_layer_6_input, LAYER_6_INPUT_SIZE * sizeof(float));

  cudaMalloc((void **)&d_layer_6_weights, LAYER_6_FILTER_SIZE * LAYER_6_FILTER_NUM * sizeof(float));
  cudaMemcpy(d_layer_6_weights, layer_6_weights, LAYER_4_FILTER_SIZE * LAYER_4_FILTER_NUM * sizeof(float), cudaMemcpyHostToDevice);
  

  // layer 7
  cudaMalloc((void **)&d_layer_7_input, LAYER_7_INPUT_SIZE * sizeof(float));

  cudaMalloc((void **)&d_layer_7_weights, LAYER_7_FILTER_SIZE * LAYER_7_FILTER_NUM * sizeof(float));
  cudaMemcpy(d_layer_7_weights, layer_7_weights, LAYER_7_FILTER_SIZE * LAYER_7_FILTER_NUM * sizeof(float), cudaMemcpyHostToDevice);
  

  // layer 8
  cudaMalloc((void **)&d_layer_8_input, LAYER_8_INPUT_SIZE * sizeof(float));

  cudaMalloc((void **)&d_layer_8_weights, LAYER_8_FILTER_SIZE * LAYER_8_FILTER_NUM * sizeof(float));
  cudaMemcpy(d_layer_8_weights, layer_8_weights, LAYER_8_FILTER_SIZE * LAYER_8_FILTER_NUM * sizeof(float), cudaMemcpyHostToDevice);




  // layer 1: 110 * 110 * 96
  dim3 conv_1_grid_dim(96, 1, 1);
  dim3 conv_1_block_dim(110, 110);

  printf("Running conv_1 ...\n");
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim, conv_1_block_dim>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110);

  dim3 pool_1_grid_dim(96, 1, 1);
  dim3 pool_1_block_dim(55, 55);
  printf("Running pool_1 ...\n");
  // stride 2, pool size 3, input_width 110, output_width 55
  run_pool<<<pool_1_grid_dim, pool_1_block_dim>>>(d_layer_1_input, d_layer_1_pooled, 2, 3, 110, 55);

  dim3 pad_1_grid_dim(96, 1, 1);
  dim3 pad_1_block_dim(55, 55);
  printf("Padding pool_1 output ...\n");
  // width 55
  run_padding<<<pad_1_grid_dim, pad_1_block_dim>>>(d_layer_1_pooled, d_layer_1_padded, 55);

  dim3 lcn_1_grid_dim(96, 1, 1);
  dim3 lcn_1_block_dim(55, 55);
  printf("Running lcn_1 ...\n");
  // width 55
  run_lcn_1<<<lcn_1_grid_dim, lcn_1_block_dim>>>(d_layer_1_padded, d_layer_1_pooled, 55);



  // layer 2: 26 * 26 * 256
  dim3 conv_2_grid_dim(256, 1, 1);
  dim3 conv_2_block_dim(26, 26);

  printf("Running conv_2 ...\n");
  // stride 2, filter size 5, channel_num 96, input_width 55, output_width 26 
  run_conv<<<conv_2_grid_dim, conv_2_block_dim>>>(d_layer_1_pooled, d_layer_2_weights, d_layer_2_input, 2, 5, 96, 55, 26);

  dim3 pool_2_grid_dim(256, 1, 1);
  dim3 pool_2_block_dim(26, 26);
  printf("Running pool_2 ...\n");
  // stride 2, pool size 3, input_width 26, output_width 13
  run_pool<<<pool_2_grid_dim, pool_2_block_dim>>>(d_layer_2_input, d_layer_2_pooled, 2, 3, 26, 13);

  dim3 pad_2_grid_dim(256, 1, 1);
  dim3 pad_2_block_dim(26, 26);
  printf("Padding pool_2 output ...\n");
  // width 26
  run_padding<<<pad_2_grid_dim, pad_2_block_dim>>>(d_layer_2_pooled, d_layer_2_padded, 26);

  dim3 lcn_2_grid_dim(256, 1, 1);
  dim3 lcn_2_block_dim(26, 26);
  printf("Running lcn_2 ...\n");
  // width 13
  run_lcn_2<<<lcn_2_grid_dim, lcn_2_block_dim>>>(d_layer_2_padded, d_layer_2_pooled, 13);



  // layer 3: 13 * 13 * 384 
  dim3 conv_3_grid_dim(384, 1, 1);
  dim3 conv_3_block_dim(13, 13);

  printf("Running conv_3 ...\n");
  // stride 1, filter size 3, channel_num 256, input_width 13,  output_width 13
  run_conv<<<conv_3_grid_dim, conv_3_block_dim>>>(d_layer_2_pooled, d_layer_3_weights, d_layer_3_input, 1, 3, 256, 13, 13);

 

  // layer 4: 13 * 13 * 384 
  dim3 conv_4_grid_dim(384, 1, 1);
  dim3 conv_4_block_dim(13, 13);

  printf("Running conv_4 ...\n");
  // stride 1, filter size 3, channel_num 384, input_width 13,  output_width 13
  run_conv<<<conv_4_grid_dim, conv_4_block_dim>>>(d_layer_3_input, d_layer_4_weights, d_layer_4_input, 1, 3, 384, 13, 13);


  // layer 5: 13 * 13 * 256
  dim3 conv_5_grid_dim(256, 1, 1);
  dim3 conv_5_block_dim(13, 13);

  printf("Running conv_5 ...\n");
  // stride 1, filter size 3, channel_num 384, input_width 13,  output_width 13
  run_conv<<<conv_5_grid_dim, conv_5_block_dim>>>(d_layer_4_input, d_layer_5_weights, d_layer_5_input, 1, 3, 384, 13, 13);

  dim3 pool_5_grid_dim(256, 1, 1);
  dim3 pool_5_block_dim(13, 13);
  printf("Running pool_5 ...\n");
  // stride 2, pool size 3, input_width 13, output_width 6
  run_pool<<<pool_5_grid_dim, pool_5_block_dim>>>(d_layer_5_input, d_layer_5_pooled, 2, 3, 13, 6);



  // layer 6: 1 * 1 * 4096
  dim3 fc_6_grid_dim(4096, 1, 1);
  dim3 fc_6_block_dim(1, 1);

  printf("Running fc_6 ...\n");
  // stride 1, filter size 1, channel_num 256, input_width 6,  output_width 1
  run_conv<<<fc_6_grid_dim, fc_6_block_dim>>>(d_layer_5_pooled, d_layer_6_weights, d_layer_6_input, 1, 1, 256, 6, 1);



  // layer 7: 1 * 1 * 4096
  dim3 fc_7_grid_dim(4096, 1, 1);
  dim3 fc_7_block_dim(1, 1);

  printf("Running fc_7 ...\n");
  // stride 1, filter size 1, channel_num 4096, input_width 1,  output_width 1
  run_conv<<<fc_7_grid_dim, fc_7_block_dim>>>(d_layer_6_input, d_layer_7_weights, d_layer_7_input, 1, 1, 4096, 1, 1);



  // layer 8: 1 * 1 * 1000
  dim3 fc_8_grid_dim(1000, 1, 1);
  dim3 fc_8_block_dim(1, 1);

  printf("Running fc_8 ...\n");
  // stride 1, filter size 1, channel_num 4096, input_width 1,  output_width 1
  run_conv<<<fc_8_grid_dim, fc_8_block_dim>>>(d_layer_7_input, d_layer_8_weights, d_layer_8_input, 1, 1, 4096, 1, 1);

  // extra relu

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