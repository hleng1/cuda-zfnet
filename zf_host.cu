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
const int LAYER_5_FILTER_SIZE = 3 * 3 * 384;
const int LAYER_5_FILTER_NUM = 256;

const int LAYER_6_INPUT_SIZE = 1 * 1 * 4096;
const int LAYER_6_FILTER_SIZE = 6 * 6 * 256;
const int LAYER_6_FILTER_NUM = 4096;

const int LAYER_7_INPUT_SIZE = 1 * 1 * 4096;
const int LAYER_7_FILTER_SIZE = 1 * 1 * 4096;
const int LAYER_7_FILTER_NUM = 4096;

const int LAYER_8_INPUT_SIZE = 1 * 1 * 1000;
const int LAYER_8_FILTER_SIZE = 1 * 1 * 4096;
const int LAYER_8_FILTER_NUM = 1000;

const int OUTPUT_SIZE = 1000;

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
  float *output_array;

  input_array = (float *)malloc(INPUT_SIZE * sizeof(float));

  layer_1_weights = (float *)malloc(LAYER_1_FILTER_SIZE * LAYER_1_FILTER_NUM * sizeof(float));
  layer_2_weights = (float *)malloc(LAYER_2_FILTER_SIZE * LAYER_2_FILTER_NUM * sizeof(float));
  layer_3_weights = (float *)malloc(LAYER_3_FILTER_SIZE * LAYER_3_FILTER_NUM * sizeof(float));
  layer_4_weights = (float *)malloc(LAYER_4_FILTER_SIZE * LAYER_4_FILTER_NUM * sizeof(float));
  layer_5_weights = (float *)malloc(LAYER_5_FILTER_SIZE * LAYER_5_FILTER_NUM * sizeof(float));
  layer_6_weights = (float *)malloc(LAYER_6_FILTER_SIZE * LAYER_6_FILTER_NUM * sizeof(float));
  layer_7_weights = (float *)malloc(LAYER_7_FILTER_SIZE * LAYER_7_FILTER_NUM * sizeof(float));
  layer_8_weights = (float *)malloc(LAYER_8_FILTER_SIZE * LAYER_8_FILTER_NUM * sizeof(float));

  output_array = (float *)malloc(OUTPUT_SIZE * sizeof(float));

  read_file("data/input.txt", input_array);


  printf("Reading layer1 weights ...\n");
  read_file("data/layer1.txt", layer_1_weights);
  printf("Reading layer2 weights ...\n");
  read_file("data/layer2.txt", layer_2_weights);
  printf("Reading layer3 weights ...\n");
  read_file("data/layer3.txt", layer_3_weights);
  printf("Reading layer4 weights ...\n");
  read_file("data/layer4.txt", layer_4_weights);
  printf("Reading layer5 weights ...\n");
  read_file("data/layer5.txt", layer_5_weights);
  printf("Reading layer6 weights ...\n");
  read_file("data/layer6.txt", layer_6_weights);
  printf("Reading layer7 weights ...\n");
  read_file("data/layer7.txt", layer_7_weights);
  printf("Reading layer8 weights ...\n");
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
  float *d_layer_8_weights;

  cudaError_t err_code;


  // input
  cudaMalloc((void **)&d_input, INPUT_SIZE * sizeof(float));
  cudaMemcpy(d_input, input_array, INPUT_SIZE* sizeof(float), cudaMemcpyHostToDevice);

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
  cudaMalloc((void **)&d_layer_5_pooled, LAYER_5_POOLED_SIZE * sizeof(float));


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
  printf("Running conv_1 ...\n");
  dim3 conv_1_grid_dim00(96, 1, 1);
  dim3 conv_1_block_dim00(32, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim00, conv_1_block_dim00>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 0, 0);


  dim3 conv_1_grid_dim01(96, 1, 1);
  dim3 conv_1_block_dim01(32, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim01, conv_1_block_dim01>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 0, 32);

  dim3 conv_1_grid_dim02(96, 1, 1);
  dim3 conv_1_block_dim02(32, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim02, conv_1_block_dim02>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 0, 64);

  dim3 conv_1_grid_dim03(96, 1, 1);
  dim3 conv_1_block_dim03(32, 14);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim03, conv_1_block_dim03>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 0, 96);

  dim3 conv_1_grid_dim10(96, 1, 1);
  dim3 conv_1_block_dim10(32, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim10, conv_1_block_dim10>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 32, 0);

  dim3 conv_1_grid_dim11(96, 1, 1);
  dim3 conv_1_block_dim11(32, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim11, conv_1_block_dim11>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 32, 32);

  dim3 conv_1_grid_dim12(96, 1, 1);
  dim3 conv_1_block_dim12(32, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim12, conv_1_block_dim12>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 32, 64);

  dim3 conv_1_grid_dim13(96, 1, 1);
  dim3 conv_1_block_dim13(32, 14);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim13, conv_1_block_dim13>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 32, 96);

  dim3 conv_1_grid_dim20(96, 1, 1);
  dim3 conv_1_block_dim20(32, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim20, conv_1_block_dim20>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 64, 0);

  dim3 conv_1_grid_dim21(96, 1, 1);
  dim3 conv_1_block_dim21(32, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim21, conv_1_block_dim21>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 64, 32);

  dim3 conv_1_grid_dim22(96, 1, 1);
  dim3 conv_1_block_dim22(32, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim22, conv_1_block_dim22>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 64, 64);

  dim3 conv_1_grid_dim23(96, 1, 1);
  dim3 conv_1_block_dim23(32, 14);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim23, conv_1_block_dim23>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 64, 96);

  dim3 conv_1_grid_dim30(96, 1, 1);
  dim3 conv_1_block_dim30(14, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim30, conv_1_block_dim30>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 96, 0);

  dim3 conv_1_grid_dim31(96, 1, 1);
  dim3 conv_1_block_dim31(14, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim31, conv_1_block_dim31>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 96, 32);

  dim3 conv_1_grid_dim32(96, 1, 1);
  dim3 conv_1_block_dim32(14, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim32, conv_1_block_dim32>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 96, 64);

  dim3 conv_1_grid_dim33(96, 1, 1);
  dim3 conv_1_block_dim33(14, 32);
  // stride 2, filter size 7, channel_num 3, input_width 224, output_width 110
  run_conv<<<conv_1_grid_dim33, conv_1_block_dim33>>>(d_input, d_layer_1_weights, d_layer_1_input, 2, 7, 3, 224, 110, 96, 96);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("conv error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }

  printf("Running pool_1 ...\n");
  dim3 pool_1_grid_dim00(96, 1, 1);
  dim3 pool_1_block_dim00(32, 32);
  // stride 2, pool size 3, input_width 110, output_width 55
  run_pool<<<pool_1_grid_dim00, pool_1_block_dim00>>>(d_layer_1_input, d_layer_1_pooled, 2, 3, 110, 55, 0, 0);

  dim3 pool_1_grid_dim01(96, 1, 1);
  dim3 pool_1_block_dim01(32, 23);
  // stride 2, pool size 3, input_width 110, output_width 55
  run_pool<<<pool_1_grid_dim01, pool_1_block_dim01>>>(d_layer_1_input, d_layer_1_pooled, 2, 3, 110, 55, 0, 32);

  dim3 pool_1_grid_dim10(96, 1, 1);
  dim3 pool_1_block_dim10(23, 32);
  // stride 2, pool size 3, input_width 110, output_width 55
  run_pool<<<pool_1_grid_dim10, pool_1_block_dim10>>>(d_layer_1_input, d_layer_1_pooled, 2, 3, 110, 55, 32, 0);

  dim3 pool_1_grid_dim11(96, 1, 1);
  dim3 pool_1_block_dim11(23, 23);
  // stride 2, pool size 3, input_width 110, output_width 55
  run_pool<<<pool_1_grid_dim11, pool_1_block_dim11>>>(d_layer_1_input, d_layer_1_pooled, 2, 3, 110, 55, 32, 32);

  printf("Padding pool_1 output ...\n");
  dim3 pad_1_grid_dim00(96, 1, 1);
  dim3 pad_1_block_dim00(32, 32);
  // width 55
  run_padding<<<pad_1_grid_dim00, pad_1_block_dim00>>>(d_layer_1_pooled, d_layer_1_padded, 55, 0, 0);

  dim3 pad_1_grid_dim01(96, 1, 1);
  dim3 pad_1_block_dim01(32, 23);
  // width 55
  run_padding<<<pad_1_grid_dim01, pad_1_block_dim01>>>(d_layer_1_pooled, d_layer_1_padded, 55, 0 ,32);

  dim3 pad_1_grid_dim10(96, 1, 1);
  dim3 pad_1_block_dim10(23, 32);
  // width 55
  run_padding<<<pad_1_grid_dim10, pad_1_block_dim10>>>(d_layer_1_pooled, d_layer_1_padded, 55, 32, 0);

  dim3 pad_1_grid_dim11(96, 1, 1);
  dim3 pad_1_block_dim11(23, 23);
  // width 55
  run_padding<<<pad_1_grid_dim11, pad_1_block_dim11>>>(d_layer_1_pooled, d_layer_1_padded, 55, 32, 32);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("pooling error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }

  printf("Running lcn_1 ...\n");
  dim3 lcn_1_grid_dim00(96, 1, 1);
  dim3 lcn_1_block_dim00(32, 32);
  // width 55
  run_lcn<<<lcn_1_grid_dim00, lcn_1_block_dim00>>>(d_layer_1_padded, d_layer_1_pooled, 55, 0, 0);

  dim3 lcn_1_grid_dim01(96, 1, 1);
  dim3 lcn_1_block_dim01(32, 23);
  // width 55
  run_lcn<<<lcn_1_grid_dim01, lcn_1_block_dim01>>>(d_layer_1_padded, d_layer_1_pooled, 55, 0, 32);

  dim3 lcn_1_grid_dim10(96, 1, 1);
  dim3 lcn_1_block_dim10(23, 32);
  // width 55
  run_lcn<<<lcn_1_grid_dim10, lcn_1_block_dim10>>>(d_layer_1_padded, d_layer_1_pooled, 55, 32, 0);

  dim3 lcn_1_grid_dim11(96, 1, 1);
  dim3 lcn_1_block_dim11(23, 23);
  // width 55
  run_lcn<<<lcn_1_grid_dim11, lcn_1_block_dim11>>>(d_layer_1_padded, d_layer_1_pooled, 55, 32, 32);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("lcn error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }


  // layer 2: 26 * 26 * 256
  dim3 conv_2_grid_dim(256, 1, 1);
  dim3 conv_2_block_dim(26, 26);

  printf("Running conv_2 ...\n");
  // stride 2, filter size 5, channel_num 96, input_width 55, output_width 26
  run_conv<<<conv_2_grid_dim, conv_2_block_dim>>>(d_layer_1_pooled, d_layer_2_weights, d_layer_2_input, 2, 5, 96, 55, 26, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("conv2 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }

  dim3 pool_2_grid_dim(256, 1, 1);
  dim3 pool_2_block_dim(26, 26);
  printf("Running pool_2 ...\n");
  // stride 2, pool size 3, input_width 26, output_width 13
  run_pool<<<pool_2_grid_dim, pool_2_block_dim>>>(d_layer_2_input, d_layer_2_pooled, 2, 3, 26, 13, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("pool2 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }

  dim3 pad_2_grid_dim(256, 1, 1);
  dim3 pad_2_block_dim(26, 26);
  printf("Padding pool_2 output ...\n");
  // width 26
  run_padding<<<pad_2_grid_dim, pad_2_block_dim>>>(d_layer_2_pooled, d_layer_2_padded, 26, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("padding2 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }

  dim3 lcn_2_grid_dim(256, 1, 1);
  dim3 lcn_2_block_dim(26, 26);
  printf("Running lcn_2 ...\n");
  // width 13
  run_lcn<<<lcn_2_grid_dim, lcn_2_block_dim>>>(d_layer_2_padded, d_layer_2_pooled, 13, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("lcn2 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }


  // layer 3: 13 * 13 * 384
  dim3 conv_3_grid_dim(384, 1, 1);
  dim3 conv_3_block_dim(13, 13);

  printf("Running conv_3 ...\n");
  // stride 1, filter size 3, channel_num 256, input_width 13,  output_width 13
  run_conv<<<conv_3_grid_dim, conv_3_block_dim>>>(d_layer_2_pooled, d_layer_3_weights, d_layer_3_input, 1, 3, 256, 13, 13, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("conv3 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }


  // layer 4: 13 * 13 * 384
  dim3 conv_4_grid_dim(384, 1, 1);
  dim3 conv_4_block_dim(13, 13);

  printf("Running conv_4 ...\n");
  // stride 1, filter size 3, channel_num 384, input_width 13,  output_width 13
  run_conv<<<conv_4_grid_dim, conv_4_block_dim>>>(d_layer_3_input, d_layer_4_weights, d_layer_4_input, 1, 3, 384, 13, 13, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("conv4 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }

  // layer 5: 13 * 13 * 256
  dim3 conv_5_grid_dim(256, 1, 1);
  dim3 conv_5_block_dim(13, 13);

  printf("Running conv_5 ...\n");
  // stride 1, filter size 3, channel_num 384, input_width 13,  output_width 13
  run_conv<<<conv_5_grid_dim, conv_5_block_dim>>>(d_layer_4_input, d_layer_5_weights, d_layer_5_input, 1, 3, 384, 13, 13, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("conv5 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }

  dim3 pool_5_grid_dim(256, 1, 1);
  dim3 pool_5_block_dim(13, 13);
  printf("Running pool_5 ...\n");
  // stride 2, pool size 3, input_width 13, output_width 6
  run_pool<<<pool_5_grid_dim, pool_5_block_dim>>>(d_layer_5_input, d_layer_5_pooled, 2, 3, 13, 6, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("pool5 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }


  // layer 6: 1 * 1 * 4096
  dim3 fc_6_grid_dim(4096, 1, 1);
  dim3 fc_6_block_dim(1, 1);

  printf("Running fc_6 ...\n");
  // stride 1, filter size 1, channel_num 256, input_width 6,  output_width 1
  run_conv<<<fc_6_grid_dim, fc_6_block_dim>>>(d_layer_5_pooled, d_layer_6_weights, d_layer_6_input, 1, 1, 256, 6, 1, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("fc6 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }


  // layer 7: 1 * 1 * 4096
  dim3 fc_7_grid_dim(4096, 1, 1);
  dim3 fc_7_block_dim(1, 1);

  printf("Running fc_7 ...\n");
  // stride 1, filter size 1, channel_num 4096, input_width 1,  output_width 1
  run_conv<<<fc_7_grid_dim, fc_7_block_dim>>>(d_layer_6_input, d_layer_7_weights, d_layer_7_input, 1, 1, 4096, 1, 1, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("fc7 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }


  // layer 8: 1 * 1 * 1000
  dim3 fc_8_grid_dim(1000, 1, 1);
  dim3 fc_8_block_dim(1, 1);

  printf("Running fc_8 ...\n");
  // stride 1, filter size 1, channel_num 4096, input_width 1,  output_width 1
  run_fc_8<<<fc_8_grid_dim, fc_8_block_dim>>>(d_layer_7_input, d_layer_8_weights, d_layer_8_input, 1, 1, 4096, 1, 1, 0, 0);

  err_code = cudaGetLastError();
  if (err_code != cudaSuccess) {
    printf("fc8 error: %s\n", cudaGetErrorString(err_code));
    exit(EXIT_FAILURE);
  }

  // extra relu

  cudaMemcpy(output_array, d_layer_8_input, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  float max = 0;
  int output_index;
  for (int i = 0; i < 1000; i++) {
    // printf("%f\n", (output_array[i]));
    if (max < output_array[i]) {
      max = output_array[i];
      output_index = i;
    }
  }
  printf("Index: %d\n", output_index);
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
    // printf("Read %zu\n", len);
    dest_array[count++] = atof(line);
  }
  free(line);
  fclose(fp);
}
