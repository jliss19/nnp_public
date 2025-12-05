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

#define THREADS_PER_BLOCK 256
#define EPSILON 1e-8f

__global__ void forward_layer_relu_kernel(const float *weights,
										  const float *bias,
										  const float *input,
										  float *pre_activation,
										  float *activation,
										  int in_dim,
										  int out_dim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= out_dim) {
		return;
	}

	float sum = bias[idx];
	for (int i = 0; i < in_dim; ++i) {
		sum += input[i] * weights[i * out_dim + idx];
	}

	if (pre_activation) {
		pre_activation[idx] = sum;
	}
	activation[idx] = sum > 0.0f ? sum : 0.0f;
}

__global__ void forward_layer_linear_kernel(const float *weights,
											const float *bias,
											const float *input,
											float *output,
											int in_dim,
											int out_dim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= out_dim) {
		return;
	}

	float sum = bias[idx];
	for (int i = 0; i < in_dim; ++i) {
		sum += input[i] * weights[i * out_dim + idx];
	}
	output[idx] = sum;
}

__global__ void softmax_kernel(const float *input,
							   float *output,
							   int len) {
	extern __shared__ float shared[];
	float *exps = shared;
	float *stats = shared + len; // stats[0]=max, stats[1]=sum

	int tid = threadIdx.x;

	if (tid == 0) {
		float max_val = input[0];
		for (int i = 1; i < len; ++i) {
			if (input[i] > max_val) {
				max_val = input[i];
			}
		}
		stats[0] = max_val;
	}
	__syncthreads();

	if (tid < len) {
		float val = expf(input[tid] - stats[0]);
		exps[tid] = val;
	}
	__syncthreads();

	if (tid == 0) {
		float sum = 0.0f;
		for (int i = 0; i < len; ++i) {
			sum += exps[i];
		}
		stats[1] = sum;
	}
	__syncthreads();

	if (tid < len) {
		float sum = stats[1] + EPSILON;
		output[tid] = exps[tid] / sum;
	}
}

__global__ void delta_output_kernel(const float *labels,
									const float *predictions,
									float *delta,
									int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
		delta[idx] = labels[idx] - predictions[idx];
	}
}

__global__ void backprop_hidden_kernel(const float *weights_next,
									   const float *delta_next,
									   const float *activation,
									   float *delta_current,
									   int current_size,
									   int next_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= current_size) {
		return;
	}

	float err = 0.0f;
	for (int k = 0; k < next_size; ++k) {
		err += delta_next[k] * weights_next[idx * next_size + k];
	}
	delta_current[idx] = activation[idx] > 0.0f ? err : 0.0f;
}

__global__ void update_weights_kernel(const float *activation,
									  const float *delta,
									  float *weights,
									  int in_dim,
									  int out_dim,
									  float lr) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = in_dim * out_dim;
	if (idx >= total) {
		return;
	}

	int row = idx / out_dim;
	int col = idx % out_dim;
	weights[row * out_dim + col] += lr * delta[col] * activation[row];
}

__global__ void update_bias_kernel(float *bias,
								   const float *delta,
								   int size,
								   float lr) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		bias[idx] += lr * delta[idx];
	}
}

cudaError_t launch_forward_layer_relu(const float *weights,
									  const float *bias,
									  const float *input,
									  float *pre_activation,
									  float *activation,
									  int in_dim,
									  int out_dim) {
	int blocks = (out_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	forward_layer_relu_kernel<<<blocks, THREADS_PER_BLOCK>>>(weights,
															 bias,
															 input,
															 pre_activation,
															 activation,
															 in_dim,
															 out_dim);
	return cudaGetLastError();
}

cudaError_t launch_forward_layer_linear(const float *weights,
										const float *bias,
										const float *input,
										float *output,
										int in_dim,
										int out_dim) {
	int blocks = (out_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	forward_layer_linear_kernel<<<blocks, THREADS_PER_BLOCK>>>(weights,
															   bias,
															   input,
															   output,
															   in_dim,
															   out_dim);
	return cudaGetLastError();
}

cudaError_t launch_softmax(const float *input,
						   float *output,
						   int len) {
	int threads = len <= 32 ? 32 : 64;
	size_t shared_size = (static_cast<size_t>(len) + 2) * sizeof(float);
	softmax_kernel<<<1, threads, shared_size>>>(input, output, len);
	return cudaGetLastError();
}

cudaError_t launch_delta_output(const float *labels,
								const float *predictions,
								float *delta,
								int len) {
	int blocks = (len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	delta_output_kernel<<<blocks, THREADS_PER_BLOCK>>>(labels,
													   predictions,
													   delta,
													   len);
	return cudaGetLastError();
}

cudaError_t launch_backprop_hidden(const float *weights_next,
								   const float *delta_next,
								   const float *activation,
								   float *delta_current,
								   int current_size,
								   int next_size) {
	int blocks = (current_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	backprop_hidden_kernel<<<blocks, THREADS_PER_BLOCK>>>(weights_next,
														  delta_next,
														  activation,
														  delta_current,
														  current_size,
														  next_size);
	return cudaGetLastError();
}

cudaError_t launch_update_weights(const float *activation,
								  const float *delta,
								  float *weights,
								  int in_dim,
								  int out_dim,
								  float lr) {
	int total = in_dim * out_dim;
	int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	update_weights_kernel<<<blocks, THREADS_PER_BLOCK>>>(activation,
														 delta,
														 weights,
														 in_dim,
														 out_dim,
														 lr);
	return cudaGetLastError();
}

cudaError_t launch_update_bias(float *bias,
							   const float *delta,
							   int size,
							   float lr) {
	int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	update_bias_kernel<<<blocks, THREADS_PER_BLOCK>>>(bias,
													  delta,
													  size,
													  lr);
	return cudaGetLastError();
}
