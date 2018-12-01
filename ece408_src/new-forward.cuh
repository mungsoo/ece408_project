
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define CUDA_MAX_NUM_THREADS 1024
#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define TILE_SIZE 32.0
#define TILE_SIZE_ 32
//__constant__ float kernel[]
__constant__ float deviceKernel[14112];

__global__ void unroll_Kernel(const int C, const int H, const int W, const int K, float *X, float *X_unrolled)
{
    int c, s, h_out, w_out, w_unroll, h_unroll, p, q;
    int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    if(t < C * W_unroll)
    {
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        w_unroll = s;
        #pragma unroll
        for(p = 0;p < K;p++)
        {
            #pragma unroll
            for(q = 0;q < K;q++)
            {
                h_unroll = c * K * K + p * K + q;
                X_unrolled[h_unroll * W_unroll + w_unroll] = X[c * H * W + (h_out + p) * W + w_out + q];
            }
        }
    }

}

__global__ void gemm_Kernel(const int M, const int H_unroll, const int W_unroll, float *X_unrolled, float *y)
{
    __shared__ float input[TILE_SIZE_ * TILE_SIZE_];
    __shared__ float localkernel[TILE_SIZE_ * TILE_SIZE_];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by * TILE_SIZE + ty, col = bx * TILE_SIZE + tx;
    float result = 0;

    #pragma unroll
    for(int ph = 0;ph < ceil(H_unroll / TILE_SIZE);ph++)
    {
        if(ph * TILE_SIZE + ty < H_unroll && col < W_unroll)
            input[ty * TILE_SIZE_ + tx] = X_unrolled[(ph * TILE_SIZE_ + ty) * W_unroll + col];
        else
            input[ty * TILE_SIZE_ + tx] = 0;
        if(row < M && ph * TILE_SIZE + tx < H_unroll)
            localkernel[ty * TILE_SIZE_ + tx] = deviceKernel[row * H_unroll + ph * TILE_SIZE_ + tx];
        else
            localkernel[ty * TILE_SIZE_ + tx] = 0;

        __syncthreads();
        #pragma unroll
        for(int i = 0;i < TILE_SIZE;i++)
            result += localkernel[ty * TILE_SIZE_ + i] * input[i * TILE_SIZE_ + tx];
        __syncthreads();

    }
    if(row < M && col < W_unroll)
        y[row * W_unroll + col] = result;
    
}
/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[2];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out * W_out;
    const int H_unroll = C * K * K;
    // const int block_width = TILE_SIZE + K - 1;
    float *X_unrolled;
    cudaMalloc((void **)&X_unrolled, B / 4 * W_unroll * H_unroll * sizeof(float));
    // ...
    cudaStream_t s = y.stream_->stream_;
    cudaStream_t stream[B/4];
    for(int i = 0;i < B / 4;i++)
        cudaStreamCreate(&stream[i]);
    cudaMemcpyToSymbol(deviceKernel, w.dptr_, sizeof(float) * K * K * M * C, 0, cudaMemcpyHostToDevice);
  
    #pragma unroll
    for(int i = 0;i < 4;i++)
    {
        #pragma unroll 100
        for(int j = 0;j < B / 4;j++)
        {
            int n = i * (B / 4) + j;
            cudaStreamCreate(&stream[j]); 
            unroll_Kernel<<<ceil((C * H_out * W_out) * 1.0 /CUDA_MAX_NUM_THREADS), CUDA_MAX_NUM_THREADS, 0, stream[j]>>>(C, H, W, K ,&(((float *)x.dptr_)[n * C * H * W]), &X_unrolled[j * H_unroll * W_unroll]);
            // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
            dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
            dim3 dimGrid(ceil(W_unroll / TILE_SIZE), ceil(M / TILE_SIZE), 1);
            gemm_Kernel<<<dimGrid, dimBlock, 0, stream[j]>>>(M, H_unroll, W_unroll, &X_unrolled[j * H_unroll * W_unroll], &(((float *)y.dptr_)[n * M * W_unroll]));
            // MSHADOW_CUDA_CALL(cudaDeviceSynchronize()); 
        }
    }

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);
    // dim3 dimGrid(ceil(W_out / TILE_SIZE), ceil(H_out / TILE_SIZE), B);
    // dim3 dimBlock(block_width, block_width, 1);
    // Call the kernel
    // forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    // forward_kernel<<<dimGrid, dimBlock, sizeof(float) * C * block_width * block_width, s>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.

}
#undef TILE_SIZE
/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif

// #ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
// #define MXNET_OPERATOR_NEW_FORWARD_CUH_

// #include <mxnet/base.h>

// namespace mxnet
// {
// namespace op
// {

// #define TILE_SIZE 24.0
// //__constant__ float kernel[]

// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     // An example use of these macros:
//     // float a = y4d(0,0,0,0)
//     // y4d(0,0,0,0) = a
//     #define i3d(i2, i1, i0) input[(i2) * (bs * bs) + (i1) * (bs) + (i0)]
//     #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.
//     We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     */
    
//     extern __shared__ float input[];

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//     const int bx = blockIdx.x, by = blockIdx.y, b = blockIdx.z;
//     const int bs = blockDim.x;
//     const int tx = threadIdx.x, ty = threadIdx.y;
    
//     // The index of input elements.
//     // Each block deal with blockSize * blockSize input and produce TILE_SIZE * TILE_SIZE output 
//     const int w = bx * TILE_SIZE + tx;
//     const int h = by * TILE_SIZE + ty;

//     float result = 0.0;

//     if (w < W && h < H) {
//         for (int c = 0; c < C; ++c) 
//             i3d(c, ty, tx)= x4d(b, c, h, w);
//     }
//     __syncthreads();

//     if (tx < TILE_SIZE && ty < TILE_SIZE && w < W_out && h < H_out) {
//         for (int m = 0; m < M; ++m) {
//             result = 0.0;
//             for (int c = 0; c < C; ++c)
//                 for (int p = 0; p < K; ++p)
//                     for (int q = 0; q < K; ++q)
//                         // TODO   #####    Change the k4d to utilize __constant__
//                         result += i3d(c, ty + p, tx + q) * k4d(m, c, p, q);
//             y4d(b, m, h, w) = result;
//         }
//     }

    
//     #undef i3d
//     #undef y4d
//     #undef x4d
//     #undef k4d
// }

// /* 
//    This function is called by new-inl.h
//    Any code you write should be executed by this function.
//    For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
// */
// template <>
// void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
// {

//     // Use mxnet's CHECK_EQ to do assertions.
    
//     // Extract the tensor dimensions into B,M,C,H,W,K
//     const int B = x.shape_[0];
//     const int M = y.shape_[1];
//     const int C = x.shape_[1];
//     const int H = x.shape_[2];
//     const int W = x.shape_[3];
//     const int K = w.shape_[2];
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

//     const int block_width = TILE_SIZE + K - 1;
//     // ...
//     cudaStream_t s = y.stream_->stream_;
//     // Set the kernel dimensions
//     // dim3 gridDim(0);
//     // dim3 blockDim(0);
//     dim3 dimGrid(ceil(W_out / TILE_SIZE), ceil(H_out / TILE_SIZE), B);
//     dim3 dimBlock(block_width, block_width, 1);
//     // Call the kernel
//     // forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
//     forward_kernel<<<dimGrid, dimBlock, sizeof(float) * C * block_width * block_width, s>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
//     // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
//     MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

// }
// #undef TILE_SIZE
// /* 
//     This tells mxnet how to do an op when it's not a float.
//     This is not used in the ECE408 project
// */
// template <typename gpu, typename DType>
// void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
// {
//     CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
// }
// }
// }

// #endif

// #ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
// #define MXNET_OPERATOR_NEW_FORWARD_CUH_

// #include <mxnet/base.h>

// namespace mxnet
// {
// namespace op
// {

// #define TILE_SIZE 24.0
// //__constant__ float kernel[]

// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     // An example use of these macros:
//     // float a = y4d(0,0,0,0)
//     // y4d(0,0,0,0) = a
//     #define i3d(i2, i1, i0) input[(i2) * (bs * bs) + (i1) * (bs) + (i0)]
//     #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.
//     We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     */
    
//     //extern __shared__ float input[];
//     //float *x_input = &input[0];
//     //float *kernel_input = 
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//     const int H_grid = ceil(H_out / TILE_SIZE);
//     const int W_grid = ceil(W_out / TILE_SIZE);

//     const int b = blockIdx.X;
//     const int m = blockIdx.y;
//     const int h = blockIdx.z / W_grid * TILE_SIZE + threadIdx.y;
//     const int w = blockIdx.z % W_grid * TILE_SIZE + threadIdx.x;
    
//     float result = 0.0;

//     // if (w < W && h < H) {
//     //     for (int c = 0; c < C; ++c) 
//     //         i3d(0, ty, tx)= x4d(b, 0, h, w);
//     // }
//     // __syncthreads();

//     if (w < W_out && h < H_out) {
//             for (int c = 0; c < C; ++c)
//                 for (int p = 0; p < K; ++p)
//                     for (int q = 0; q < K; ++q)
//                         // TODO   #####    Change the k4d to utilize __constant__
//                         result += i3d(c, ty + p, tx + q) * k4d(m, c, p, q);
//             y4d(b, m, h, w) = result;
//         }
//     }

    
//     #undef i3d
//     #undef y4d
//     #undef x4d
//     #undef k4d
// }

// /* 
//    This function is called by new-inl.h
//    Any code you write should be executed by this function.
//    For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
// */
// template <>
// void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
// {

//     // Use mxnet's CHECK_EQ to do assertions.
    
//     // Extract the tensor dimensions into B,M,C,H,W,K
//     const int B = x.shape_[0];
//     const int M = y.shape_[1];
//     const int C = x.shape_[1];
//     const int H = x.shape_[2];
//     const int W = x.shape_[3];
//     const int K = w.shape_[2];
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//     const int H_grid = ceil(H_out / TILE_SIZE);
//     const int W_grid = ceil(W_out / TILE_SIZE);
//     const int block_width = TILE_SIZE + K - 1;
//     // ...
//     cudaStream_t s = y.stream_->stream_;
//     // Set the kernel dimensions
//     // dim3 gridDim(0);
//     // dim3 blockDim(0);
//     dim3 dimGrid(B, M, H_grid * W_grid);
//     dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
//     // Call the kernel
//     // forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
//     forward_kernel<<<dimGrid, dimBlock, 0, s>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
//     // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
//     MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

// }
// #undef TILE_SIZE
// /* 
//     This tells mxnet how to do an op when it's not a float.
//     This is not used in the ECE408 project
// */
// template <typename gpu, typename DType>
// void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
// {
//     CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
// }
// }
// }

// #endif