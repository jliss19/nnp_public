/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *
 *  CUDA kernels and launch helpers for neural network training.
 */

#include "kernels.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

#define BLOCK_SIZE 256
#define SMALL 1e-8f

__global__ void relu_layer_kernel(const float *w,
                                  const float *b,
                                  const float *x,
                                  float *pre,
                                  float *out,
                                  int in_size,
                                  int out_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= out_size) {
        return;
    }

    float sum = b[id];
    for (int i = 0; i < in_size; ++i) {
        sum += x[i] * w[i * out_size + id];
    }

    if (pre) {
        pre[id] = sum;
    }
    out[id] = sum > 0.0f ? sum : 0.0f;
}

__global__ void linear_layer_kernel(const float *w,
                                    const float *b,
                                    const float *x,
                                    float *out,
                                    int in_size,
                                    int out_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= out_size) {
        return;
    }

    float sum = b[id];
    for (int i = 0; i < in_size; ++i) {
        sum += x[i] * w[i * out_size + id];
    }
    out[id] = sum;
}

__global__ void softmax_kernel(const float *in,
                               float *out,
                               int len) {
    extern __shared__ float shared[];
    float *vals = shared;
    float *info = shared + len; // info[0] = max, info[1] = sum

    int tid = threadIdx.x;

    if (tid == 0) {
        float max_val = in[0];
        for (int i = 1; i < len; ++i) {
            if (in[i] > max_val) {
                max_val = in[i];
            }
        }
        info[0] = max_val;
    }
    __syncthreads();

    if (tid < len) {
        float val = expf(in[tid] - info[0]);
        vals[tid] = val;
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < len; ++i) {
            sum += vals[i];
        }
        info[1] = sum;
    }
    __syncthreads();

    if (tid < len) {
        float sum = info[1] + SMALL;
        out[tid] = vals[tid] / sum;
    }
}

__global__ void output_delta_kernel(const float *label,
                                    const float *pred,
                                    float *delta,
                                    int len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < len) {
        delta[id] = label[id] - pred[id];
    }
}

__global__ void hidden_delta_kernel(const float *next_w,
                                    const float *next_delta,
                                    const float *act,
                                    float *cur_delta,
                                    int cur_size,
                                    int next_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cur_size) {
        return;
    }

    float err = 0.0f;
    for (int k = 0; k < next_size; ++k) {
        err += next_delta[k] * next_w[id * next_size + k];
    }
    cur_delta[id] = act[id] > 0.0f ? err : 0.0f;
}

__global__ void weight_step_kernel(const float *act,
                                   const float *delta,
                                   float *w,
                                   int in_size,
                                   int out_size,
                                   float lr) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = in_size * out_size;
    if (id >= total) {
        return;
    }

    int row = id / out_size;
    int col = id % out_size;
    w[row * out_size + col] += lr * delta[col] * act[row];
}

__global__ void bias_step_kernel(float *b,
                                 const float *delta,
                                 int size,
                                 float lr) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        b[id] += lr * delta[id];
    }
}

cudaError_t run_relu_layer(const float *w,
                           const float *b,
                           const float *x,
                           float *pre,
                           float *out,
                           int in_size,
                           int out_size) {
    int blocks = (out_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_layer_kernel<<<blocks, BLOCK_SIZE>>>(w,
                                              b,
                                              x,
                                              pre,
                                              out,
                                              in_size,
                                              out_size);
    return cudaGetLastError();
}

cudaError_t run_linear_layer(const float *w,
                             const float *b,
                             const float *x,
                             float *out,
                             int in_size,
                             int out_size) {
    int blocks = (out_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    linear_layer_kernel<<<blocks, BLOCK_SIZE>>>(w,
                                                b,
                                                x,
                                                out,
                                                in_size,
                                                out_size);
    return cudaGetLastError();
}

cudaError_t run_softmax(const float *in,
                        float *out,
                        int len) {
    int threads = len <= 32 ? 32 : 64;
    size_t shared_bytes = (static_cast<size_t>(len) + 2) * sizeof(float);
    softmax_kernel<<<1, threads, shared_bytes>>>(in, out, len);
    return cudaGetLastError();
}

cudaError_t run_output_delta(const float *label,
                             const float *pred,
                             float *delta,
                             int len) {
    int blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    output_delta_kernel<<<blocks, BLOCK_SIZE>>>(label,
                                                pred,
                                                delta,
                                                len);
    return cudaGetLastError();
}

cudaError_t run_hidden_delta(const float *next_w,
                             const float *next_delta,
                             const float *act,
                             float *cur_delta,
                             int cur_size,
                             int next_size) {
    int blocks = (cur_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hidden_delta_kernel<<<blocks, BLOCK_SIZE>>>(next_w,
                                                next_delta,
                                                act,
                                                cur_delta,
                                                cur_size,
                                                next_size);
    return cudaGetLastError();
}

cudaError_t run_weight_step(const float *act,
                            const float *delta,
                            float *w,
                            int in_size,
                            int out_size,
                            float lr) {
    int total = in_size * out_size;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    weight_step_kernel<<<blocks, BLOCK_SIZE>>>(act,
                                               delta,
                                               w,
                                               in_size,
                                               out_size,
                                               lr);
    return cudaGetLastError();
}

cudaError_t run_bias_step(float *b,
                          const float *delta,
                          int size,
                          float lr) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bias_step_kernel<<<blocks, BLOCK_SIZE>>>(b,
                                             delta,
                                             size,
                                             lr);
    return cudaGetLastError();
}
