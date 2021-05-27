#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
//#include <cublas.h>
#include <cublas_v2.h>
// 32 64 128
#define MAX 16 // matrix length
#define BLOCKS 16
#define EPOCH 100
#pragma warning(disable:4996)

#define BLOCK_SIZE 16

__global__ void MatMulKernel(double* A, double* B, double* C);

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
  return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
  A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
  Matrix Asub;
  Asub.width = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;
  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
  return Asub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(double* A, double* B, double* C) {
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
  // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  float Cvalue = 0.0;
  // Thread row and column within Csub
  int row = threadIdx.y;
  int col = threadIdx.x;
  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    // Get sub-matrix Asub of A
    Matrix Asub = GetSubMatrix(A, blockRow, m);
    Matrix Bsub = GetSubMatrix(B, m, blockCol);
    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = GetElement(Bsub, row, col);
    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();
    // Multiply Asub and Bsub together
    for (int e = 0; e < BLOCK_SIZE; ++e)
      Cvalue += As[row][e] * Bs[e][col];
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  // Write Csub to device memory
  // Each thread writes one element
  SetElement(Csub, row, col, Cvalue);
}

void MatMul(double* A, double* B, double* C) {
  // Load A and B to device memory
  double* cuda_A = 0;
  double* cuda_B = 0;
  double* cuda_C = 0;
  cudaError_t err;

  cudaMalloc((void**)&cuda_A, MAX * MAX * sizeof(double));
  cudaMalloc((void**)&cuda_B, MAX * MAX * sizeof(double));
  cudaMalloc((void**)&cuda_C, MAX * MAX * sizeof(double));

  cudaMemcpy(cuda_A, A, MAX * MAX * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_B, B, MAX * MAX * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_C, C, MAX * MAX * sizeof(double), cudaMemcpyHostToDevice);

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(MAX / dimBlock.x, MAX / dimBlock.y);
  MatMulKernel << <dimGrid, dimBlock >> > (cuda_A, cuda_B, cuda_C);

  err = cudaThreadSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));
  // Read C from device memory
  err = cudaMemcpy(C, cuda_C, MAX * MAX * sizeof(double), cudaMemcpyDeviceToHost);
  printf("Copy C off of device: %s\n", cudaGetErrorString(err));
  // Free device memory
  cudaFree(cuda_A);
  cudaFree(cuda_B);
  cudaFree(cuda_C);
}

int main() {
  double A[MAX * MAX], B[MAX * MAX], C[MAX * MAX];

  for (int i = 0; i < MAX * MAX; i++) {
    A[i] = 1.0;
    B[i] = 1.0;
    C[i] = 0.0;
  }

  /*
  A.elements = (float*)malloc(MAX * MAX * sizeof(float));
  B.elements = (float*)malloc(MAX * MAX * sizeof(float));
  C.elements = (float*)malloc(MAX * MAX * sizeof(float));
  */

  MatMul(A, B, C);

  for (int i = 0; i < MAX * MAX; i++) {
    if ((i + 1) % MAX == 0) {
      printf("%.1lf\n", C[i]);
    }
    else {
      printf("%.1lf ", C[i]);
    }
  }

}



//int main() {
//  double A[MAX * MAX], B[MAX * MAX], C[MAX * MAX];
//  float time, time_avg = 0;
//  cudaEvent_t start, stop;
//
//  cudaEventCreate(&start); // create start event
//  cudaEventCreate(&stop); // create stop event
//  cudaEventRecord(start, 0); // record start event
//
//  for (int i = 0; i < MAX * MAX; i++) { // create A, B data
//    A[i] = 1.0;
//    B[i] = 1.0;
//    C[i] = 0.0;
//  }
//
//  for (int t = 0; t < EPOCH; t++) { // EPOCH만큼 반복하여 평균시간을 계산
//    double* cuda_A = 0;
//    double* cuda_B = 0;
//    double* cuda_C = 0;
//    cudaMalloc((void**)&cuda_A, MAX * MAX * sizeof(double));
//    cudaMalloc((void**)&cuda_B, MAX * MAX * sizeof(double));
//    cudaMalloc((void**)&cuda_C, MAX * MAX * sizeof(double));
//
//    cudaMemcpy(cuda_A, A, MAX * MAX * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(cuda_B, B, MAX * MAX * sizeof(double), cudaMemcpyHostToDevice);
//    //cudaMemcpy(cuda_C, C, MAX * MAX * sizeof(double), cudaMemcpyHostToDevice);
//
//    //dim3 dimGrid(MAX / BLOCKS, MAX / BLOCKS);
//    //dim3 dimBlock(BLOCKS, BLOCKS);
//
//    cudaEventCreate(&start); // 시간 기록
//    cudaEventCreate(&stop);
//    cudaEventRecord(start, 0);
//
//    dim3 dimGrid(ceil((double)MAX / BLOCK_SIZE), ceil((double)MAX / BLOCK_SIZE));
//    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//
//    //matrix_multiplication_with_shared_memory << < dimGrid, dimBlock >> > (cuda_C, cuda_A, cuda_B); // matrix multiplication
//    matrix_multiplication_with_shared_memory << < dimGrid, dimBlock >> > (cuda_C, cuda_A, cuda_B, MAX, MAX, MAX); // matrix multiplication
//
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&time, start, stop);
//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);
//
//    cudaMemcpy(C, cuda_C, MAX * MAX * sizeof(double), cudaMemcpyDeviceToHost);
//
//    if (t == 0) {
//      for (int i = 0; i < MAX * MAX; i++) {
//        if ((i + 1) % MAX == 0) {
//          printf("%.1lf\n", C[i]);
//        }
//        else {
//          printf("%.1lf ", C[i]);
//        }
//      }
//    }
//
//    time_avg += time;
//
//    // 메모리 해제
//    cudaFree(cuda_A);
//    cudaFree(cuda_B);
//    cudaFree(cuda_C);
//  }
//
//  printf("time : %f\n", time_avg / (float)EPOCH); // 평균 소요 시간 출력
//  return 0;
//}
