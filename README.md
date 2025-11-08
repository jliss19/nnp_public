# nnp_public
## Overview
This is a starter kit for a small neural network with 2 hidden layers.  It contains the data files for the MNIST dataset which is a series of 28x28 pixel images of hand written digits.  The code contains a serial implementation of the network as well as a working setup for a CUDA implementation (without the cuda work done, but able to be built with cuda.

## Details
The pre-parallel folder is set up so that you can use cuda calls in the file nnp.cu and place your kernel funcitons in kernels.cu/kernels.h.  The serial (and pre-parallel) versions will take several minutes to train (390 seconds on my laptop).  The CUDA version should be substantially faster.  I would expect that on Darwin, a decent cuda implementation could achieve a time of about 30-40 seconds, and an optimized one would be even faster.
