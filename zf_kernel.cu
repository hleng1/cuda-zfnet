__global__ void run_conv_1(float *d_layer_1_input, float *d_layer_1_weights, float *d_layer_2_input) {
  // stride 2, filter size 7
  float product = 0;
  int stride = 2;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      product += d_layer_1_input[3 * (i * 224 + j + tx * stride + ty * 224 * stride)] * d_layer_1_weights[i * 11 + j + bx * 11 * 11 * 3 + 11 * 11];
      product += d_layer_1_input[3 * (i * 224 + j + tx * stride + ty * 224 * stride) + 1] * d_layer_1_weights[i * 11 + j + bx * 11 * 11 * 3 + 11 * 11 * 2];
      product += d_layer_1_input[3 * (i * 224 + j + tx * stride + ty * 224 * stride) + 2] * d_layer_1_weights[i * 11 + j + bx * 11 * 11 * 3 + 11 * 11 * 3];
    }
  }
  // ReLU
  if (product < 0) {
    product = 0;
  }
  d_layer_2_input[tx + ty * 110 + bx * 110 * 110] = product;
  product = 0;
}

__global__ void run_pool_1(float *d_layer_2_input, float *d_layer_2_pooled) {
  // stride 2, pool size 3
  float max = 0, cur;
  int stride = 2;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cur = d_layer_2_input[i * 110 + j + tx * stride + ty * 110 * stride + bx * 110 * 110];
      if (max < cur) {
        max = cur;
      }
    }
  }
  d_layer_2_pooled[tx + ty * 55 + bx * 55 * 55] = max;
  max = 0;
}









