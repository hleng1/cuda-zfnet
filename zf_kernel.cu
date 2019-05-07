#include <math.h>

__global__ void run_conv_1(float *d_input, float *d_layer_1_weights, float *d_layer_1_input) {
  // stride 2, filter size 7
  float product = 0;
  int stride = 2;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      product += d_input[3 * (i * 224 + j + tx * stride + ty * 224 * stride)] * d_layer_1_weights[i * 11 + j + bx * 11 * 11 * 3 + 11 * 11];
      product += d_input[3 * (i * 224 + j + tx * stride + ty * 224 * stride) + 1] * d_layer_1_weights[i * 11 + j + bx * 11 * 11 * 3 + 11 * 11 * 2];
      product += d_input[3 * (i * 224 + j + tx * stride + ty * 224 * stride) + 2] * d_layer_1_weights[i * 11 + j + bx * 11 * 11 * 3 + 11 * 11 * 3];
    }
  }
  // ReLU
  if (product < 0) {
    product = 0;
  }
  d_layer_1_input[tx + ty * 110 + bx * 110 * 110] = product;
  product = 0;
}

__global__ void run_pool_1(float *d_layer_1_input, float *d_layer_1_pooled) {
  // stride 2, pool size 3
  float max = 0, cur;
  int stride = 2;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cur = d_layer_1_input[i * 110 + j + tx * stride + ty * 110 * stride + bx * 110 * 110];
      if (max < cur) {
        max = cur;
      }
    }
  }
  d_layer_1_pooled[tx + ty * 55 + bx * 55 * 55] = max;
  max = 0;
}

__global__ void run_padding_1(float *d_layer_1_pooled, float *d_layer_1_padded) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  d_layer_1_padded[(ty + 1) * 55 + (tx + 1) + bx * 55 * 55] = d_layer_1_pooled[ty * 55 + tx + bx * 55 * 55];
}

__global__ void run_lcn_1(float *d_layer_1_padded, float *d_layer_1_pooled) {
  // reuse d_layer_1_pooled to store the result array of layer 1
  float sum = 0;
  float mean = 0;
  float sv = 0; // standard variance
  float sd = 0; // standard deviation
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  // caculate mean of adjacent 9 pixels
  for (int p = -1; p < 1; p++) {
    for (int q = -1; q < 1; q++) {
      sum += d_layer_1_padded[(ty + p) * 55 + (tx + q) + bx * 55 * 55];
    }
  }
  mean = sum / 9;
  // calculate standard variance
  for (int p = -1; p < 1; p++) {
    for (int q = -1; q < 1; q++) {
      sv += pow(d_layer_1_padded[ty * 55 + tx + bx * 55 * 55] - mean, 2);
    }
  }
  // calculate standard deviation
  sd = sqrtf(sv/9);
  if (sd > 1) {
    d_layer_1_pooled[tx + ty * 55 + bx * 55 * 55] /= sd; 
  }
}







