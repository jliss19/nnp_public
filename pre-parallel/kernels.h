/*
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *
 *  CUDA kernel launcher declarations for neural network training.
 */

#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

cudaError_t run_relu_layer(const float *w,
                           const float *b,
                           const float *x,
                           float *pre,
                           float *out,
                           int in_size,
                           int out_size);

cudaError_t run_linear_layer(const float *w,
                             const float *b,
                             const float *x,
                             float *out,
                             int in_size,
                             int out_size);

cudaError_t run_softmax(const float *in,
                        float *out,
                        int len);

cudaError_t run_output_delta(const float *label,
                             const float *pred,
                             float *delta,
                             int len);

cudaError_t run_hidden_delta(const float *next_w,
                             const float *next_delta,
                             const float *act,
                             float *cur_delta,
                             int cur_size,
                             int next_size);

cudaError_t run_weight_step(const float *act,
                            const float *delta,
                            float *w,
                            int in_size,
                            int out_size,
                            float lr);

cudaError_t run_bias_step(float *b,
                          const float *delta,
                          int size,
                          float lr);

#endif
