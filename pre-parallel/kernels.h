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

cudaError_t launch_forward_layer_relu(const float *weights,
									  const float *bias,
									  const float *input,
									  float *pre_activation,
									  float *activation,
									  int in_dim,
									  int out_dim);

cudaError_t launch_forward_layer_linear(const float *weights,
										const float *bias,
										const float *input,
										float *output,
										int in_dim,
										int out_dim);

cudaError_t launch_softmax(const float *input,
						   float *output,
						   int len);

cudaError_t launch_delta_output(const float *labels,
								const float *predictions,
								float *delta,
								int len);

cudaError_t launch_backprop_hidden(const float *weights_next,
								   const float *delta_next,
								   const float *activation,
								   float *delta_current,
								   int current_size,
								   int next_size);

cudaError_t launch_update_weights(const float *activation,
								  const float *delta,
								  float *weights,
								  int in_dim,
								  int out_dim,
								  float lr);

cudaError_t launch_update_bias(float *bias,
							   const float *delta,
							   int size,
							   float lr);

#endif
