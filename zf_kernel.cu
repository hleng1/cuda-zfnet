#include <math.h>

__global__ void run_conv(float *d_input, float *d_layer_1_weights, float *d_layer_1_input, int stride, int filter_size, int channel_num, int input_width, int output_width, int ty_offset, int tx_offset) {
  float product = 0;
  int tx = threadIdx.x + tx_offset;
  int ty = threadIdx.y + ty_offset;
  int bx = blockIdx.x;
  for (int k = 1; k <= channel_num; k++) {
    for (int i = 0; i < filter_size; i++) {
      for (int j = 0; j < filter_size; j++) {
        product += d_input[channel_num * (i * input_width + j + tx * stride + ty * input_width * stride) + k] * d_layer_1_weights[i * filter_size + j + bx * filter_size * filter_size * channel_num + filter_size * filter_size * k];
      }
    }
  }
  // ReLU
  if (product < 0) {
    product = 0;
  }
  d_layer_1_input[tx + ty * output_width + bx * output_width * output_width] = product;
  product = 0;
}

__global__ void run_fc_8(float *d_input, float *d_layer_1_weights, float *d_layer_1_input, int stride, int filter_size, int channel_num, int input_width, int output_width, int ty_offset, int tx_offset) {
  float product = 0;
  int tx = threadIdx.x + tx_offset;
  int ty = threadIdx.y + ty_offset;
  int bx = blockIdx.x;
  for (int k = 1; k <= channel_num; k++) {
    for (int i = 0; i < filter_size; i++) {
      for (int j = 0; j < filter_size; j++) {
        product += d_input[channel_num * (i * input_width + j + tx * stride + ty * input_width * stride) + k] * d_layer_1_weights[i * filter_size + j + bx * filter_size * filter_size * channel_num + filter_size * filter_size * k];
      }
    }
  }
  // no ReLU
  d_layer_1_input[tx + ty * output_width + bx * output_width * output_width] = product;
  product = 0;
}


__global__ void run_pool(float *d_layer_1_input, float *d_layer_1_pooled, int stride, int pool_size, int input_width, int output_width, int ty_offset, int tx_offset) {
  float max = 0, cur;
  int tx = threadIdx.x + tx_offset;
  int ty = threadIdx.y + ty_offset;
  int bx = blockIdx.x;
  for (int i = 0; i < pool_size; i++) {
    for (int j = 0; j < pool_size; j++) {
      cur = d_layer_1_input[i * input_width + j + tx * stride + ty * input_width * stride + bx * input_width * input_width];
      if (max < cur) {
        max = cur;
      }
    }
  }
  d_layer_1_pooled[tx + ty * output_width + bx * output_width * output_width] = max;
  max = 0;
}

__global__ void run_padding(float *d_layer_1_pooled, float *d_layer_1_padded, int width, int ty_offset, int tx_offset) {
  int tx = threadIdx.x + tx_offset;
  int ty = threadIdx.y + ty_offset;
  int bx = blockIdx.x;
  d_layer_1_padded[(ty + 1) * width + (tx + 1) + bx * width * width] = d_layer_1_pooled[ty * width + tx + bx * width * width];
}


__global__ void run_lcn(float *d_layer_1_padded, float *d_layer_1_pooled, int width, int ty_offset, int tx_offset) {
  // reuse d_layer_1_pooled to store the result array of layer 1
  float sum = 0;
  float mean = 0;
  float sv = 0; // standard variance
  float sd = 0; // standard deviation
  int tx = threadIdx.x + tx_offset;
  int ty = threadIdx.y + ty_offset;
  int bx = blockIdx.x;
  // caculate mean of adjacent 9 pixels
  for (int p = -1; p < 1; p++) {
    for (int q = -1; q < 1; q++) {
      sum += d_layer_1_padded[(ty + p) * width + (tx + q) + bx * width * width];
    }
  }
  mean = sum / 9;
  // calculate standard variance
  for (int p = -1; p < 1; p++) {
    for (int q = -1; q < 1; q++) {
      sv += pow(d_layer_1_padded[ty * width + tx + bx * width * width] - mean, 2);
    }
  }
  // calculate standard deviation
  sd = sqrtf(sv/9);
  if (sd > 1) {
    d_layer_1_pooled[tx + ty * width + bx * width * width] /= sd; 
  }
}

