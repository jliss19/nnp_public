/*
    nnp.cu

    Created on: Nov 9, 2025
    CUDA-parallel implementation of a simple feedforward neural network for MNIST digit classification.

    Network architecture:
    - Input layer: 784 neurons (28x28 pixels)
    - Hidden layer 1: 128 neurons, ReLU activation
    - Hidden layer 2: 64 neurons, ReLU activation
    - Output layer: 10 neurons, Softmax activation

    Training:
    - Loss function: Categorical Cross-Entropy
    - Optimizer: Stochastic Gradient Descent (SGD) executed on the GPU
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"
#include "kernels.h"


/* Activation functions for relu layers
* Arguments:
*   x: input value
* Returns:
*   activated value based on ReLU function 
*/
float relu(float x) { return x > 0 ? x : 0; }

/* Derivative of ReLU activation function
* Arguments:
*   y: output value from ReLU function
* Returns:
*   derivative value
*/
float drelu(float y) { return y > 0 ? 1 : 0; }

/* Softmax activation function
* Arguments:
*   z: input array
*   out: output array to store softmax results
*   len: length of the input/output arrays
*/ 
void softmax(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

/* Initialize weights with small random values
* Arguments:
*   w: weight array to initialize
*   size: number of weights
*/
void init_weights(float *w, int size) {
    for (int i=0;i<size;i++)
        w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
}

/* CUDA error checking helper */
#define CHECK_CUDA(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(err__));                                           \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

/* Train the model using stochastic gradient descent on the GPU */
void train_model(MODEL* model){
    init_weights(model->W1, SIZE*H1); init_weights(model->b1, H1);
    init_weights(model->W2, H1*H2); init_weights(model->b2, H2);
    init_weights(model->W3, H2*CLASSES); init_weights(model->b3, CLASSES);

    // Allocate device copies of model parameters
    float *d_W1 = NULL, *d_b1 = NULL;
    float *d_W2 = NULL, *d_b2 = NULL;
    float *d_W3 = NULL, *d_b3 = NULL;
    CHECK_CUDA(cudaMalloc(&d_W1, sizeof(float) * SIZE * H1));
    CHECK_CUDA(cudaMalloc(&d_b1, sizeof(float) * H1));
    CHECK_CUDA(cudaMalloc(&d_W2, sizeof(float) * H1 * H2));
    CHECK_CUDA(cudaMalloc(&d_b2, sizeof(float) * H2));
    CHECK_CUDA(cudaMalloc(&d_W3, sizeof(float) * H2 * CLASSES));
    CHECK_CUDA(cudaMalloc(&d_b3, sizeof(float) * CLASSES));

    CHECK_CUDA(cudaMemcpy(d_W1, model->W1, sizeof(float) * SIZE * H1, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, model->b1, sizeof(float) * H1, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, model->W2, sizeof(float) * H1 * H2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, model->b2, sizeof(float) * H2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W3, model->W3, sizeof(float) * H2 * CLASSES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b3, model->b3, sizeof(float) * CLASSES, cudaMemcpyHostToDevice));

    // Allocate and copy training data to device memory
    float *d_train_data = NULL;
    float *d_train_label = NULL;
    CHECK_CUDA(cudaMalloc(&d_train_data, sizeof(float) * NUM_TRAIN * SIZE));
    CHECK_CUDA(cudaMalloc(&d_train_label, sizeof(float) * NUM_TRAIN * CLASSES));
    CHECK_CUDA(cudaMemcpy(d_train_data, train_data, sizeof(float) * NUM_TRAIN * SIZE,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_train_label, train_label, sizeof(float) * NUM_TRAIN * CLASSES,
                          cudaMemcpyHostToDevice));

    // Allocate workspace on the device
    float *d_h1a = NULL;
    float *d_h2a = NULL;
    float *d_out = NULL, *d_outa = NULL;
    float *d_delta1 = NULL, *d_delta2 = NULL, *d_delta3 = NULL;

    CHECK_CUDA(cudaMalloc(&d_h1a, sizeof(float) * H1));
    CHECK_CUDA(cudaMalloc(&d_h2a, sizeof(float) * H2));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * CLASSES));
    CHECK_CUDA(cudaMalloc(&d_outa, sizeof(float) * CLASSES));
    CHECK_CUDA(cudaMalloc(&d_delta1, sizeof(float) * H1));
    CHECK_CUDA(cudaMalloc(&d_delta2, sizeof(float) * H2));
    CHECK_CUDA(cudaMalloc(&d_delta3, sizeof(float) * CLASSES));

    float host_outa[CLASSES];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0.0;

        for (int n = 0; n < NUM_TRAIN; n++) {
            const float *d_x = d_train_data + (size_t)n * SIZE;
            const float *d_y = d_train_label + (size_t)n * CLASSES;

            CHECK_CUDA(launch_forward_layer_relu(d_W1, d_b1, d_x, NULL, d_h1a, SIZE, H1));
            CHECK_CUDA(launch_forward_layer_relu(d_W2, d_b2, d_h1a, NULL, d_h2a, H1, H2));
            CHECK_CUDA(launch_forward_layer_linear(d_W3, d_b3, d_h2a, d_out, H2, CLASSES));
            CHECK_CUDA(launch_softmax(d_out, d_outa, CLASSES));

            CHECK_CUDA(cudaMemcpy(host_outa, d_outa, sizeof(float) * CLASSES,
                                  cudaMemcpyDeviceToHost));
            for (int k = 0; k < CLASSES; ++k) {
                float target = train_label[n][k];
                loss -= target * logf(host_outa[k] + 1e-8f);
            }

            CHECK_CUDA(launch_delta_output(d_y, d_outa, d_delta3, CLASSES));
            CHECK_CUDA(launch_backprop_hidden(d_W3, d_delta3, d_h2a, d_delta2, H2, CLASSES));
            CHECK_CUDA(launch_backprop_hidden(d_W2, d_delta2, d_h1a, d_delta1, H1, H2));

            CHECK_CUDA(launch_update_weights(d_h2a, d_delta3, d_W3, H2, CLASSES, LR));
            CHECK_CUDA(launch_update_bias(d_b3, d_delta3, CLASSES, LR));

            CHECK_CUDA(launch_update_weights(d_h1a, d_delta2, d_W2, H1, H2, LR));
            CHECK_CUDA(launch_update_bias(d_b2, d_delta2, H2, LR));

            CHECK_CUDA(launch_update_weights(d_x, d_delta1, d_W1, SIZE, H1, LR));
            CHECK_CUDA(launch_update_bias(d_b1, d_delta1, H1, LR));
        }

        printf("Epoch %d, Loss=%.4f\n", epoch, loss / (double)NUM_TRAIN);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(model->W1, d_W1, sizeof(float) * SIZE * H1, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(model->b1, d_b1, sizeof(float) * H1, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(model->W2, d_W2, sizeof(float) * H1 * H2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(model->b2, d_b2, sizeof(float) * H2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(model->W3, d_W3, sizeof(float) * H2 * CLASSES, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(model->b3, d_b3, sizeof(float) * CLASSES, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_delta3));
    CHECK_CUDA(cudaFree(d_delta2));
    CHECK_CUDA(cudaFree(d_delta1));
    CHECK_CUDA(cudaFree(d_outa));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_h2a));
    CHECK_CUDA(cudaFree(d_h1a));
    CHECK_CUDA(cudaFree(d_train_label));
    CHECK_CUDA(cudaFree(d_train_data));
    CHECK_CUDA(cudaFree(d_b3));
    CHECK_CUDA(cudaFree(d_W3));
    CHECK_CUDA(cudaFree(d_b2));
    CHECK_CUDA(cudaFree(d_W2));
    CHECK_CUDA(cudaFree(d_b1));
    CHECK_CUDA(cudaFree(d_W1));
}

/* Save the trained model to a binary file
* Arguments:
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None
*/
void save_model(MODEL* model){
	FILE *f = fopen("model.bin", "wb");
	fwrite(model->W1, sizeof(float), SIZE*H1, f);
	fwrite(model->b1, sizeof(float), H1, f);
	fwrite(model->W2, sizeof(float), H1*H2, f);
	fwrite(model->b2, sizeof(float), H2, f);
	fwrite(model->W3, sizeof(float), H2*CLASSES, f);
	fwrite(model->b3, sizeof(float), CLASSES,f);
	fclose(f);
}

/* Load the trained model from a binary file
* Arguments:
*   model (out): pointer to the MODEL structure to populate with loaded weights and biases
* Returns:
*   None
*/
void load_model(MODEL* model){
	FILE *f = fopen("model.bin", "rb");
	fread(model->W1, sizeof(float), SIZE*H1, f);
	fread(model->b1, sizeof(float), H1, f);
	fread(model->W2, sizeof(float), H1*H2, f);
	fread(model->b2, sizeof(float), H2, f);
	fread(model->W3, sizeof(float), H2*CLASSES, f);
	fread(model->b3, sizeof(float), CLASSES, f);
	fclose(f);
}

/* Predict the class of a given input image
* Arguments:
*   x: input image array (flattened 28x28 pixels)
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None (prints predicted class and confidence)
*/
void predict(float *x, MODEL* model){
    float h1[H1], h1a[H1], h2[H2], h2a[H2], out[CLASSES], outa[CLASSES];

    // forward pass
    for (int j=0;j<H1;j++){ h1[j]=model->b1[j]; for(int i=0;i<SIZE;i++) h1[j]+=x[i]*model->W1[i*H1+j]; h1a[j]=relu(h1[j]); }
    for (int j=0;j<H2;j++){ h2[j]=model->b2[j]; for(int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j]; h2a[j]=relu(h2[j]); }
    for (int k=0;k<CLASSES;k++){ out[k]=model->b3[k]; for(int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k]; }
    softmax(out,outa,CLASSES);

    // print predicted class
    int pred=0; float max=outa[0];
    for(int k=1;k<CLASSES;k++) if(outa[k]>max){ max=outa[k]; pred=k; }
    printf("Predicted digit: %d (confidence %.2f)\n", pred, max);
}


