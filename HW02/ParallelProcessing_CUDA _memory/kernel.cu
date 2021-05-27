#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
//#include <cublas.h>
#include <cublas_v2.h>
// 32 64 128
#define MAX 16 // matrix length
#define BLOCKS 16

#define INDEX2ROW(_index,_width)	(int)((_index)/(_width))
#define INDEX2COL(_index,_width)	((_index)%(_width))
#define ID2INDEX(_row,_col, _width) (((_row)*(_width))+(_col))

#define BLOCK_SIZE 16
#define LOOP_I(_size) for(int i = 0 ; i < _size; i++)
#define KERNEL_MUL(_a,_b) __fmul_rn(_a,_b)

#define EPOCH 100
#pragma warning(disable:4996)

//__global__ void matrix_multiplication_with_shared_memory(double* C, double* A, double* B) {
//  int y = threadIdx.y;
//  int x = threadIdx.x;
//  __shared__ double shmem[MAX];
//
//  int k = threadIdx.x;
//  double sum = 0.0;
//
//  shmem[k] = A[MAX * y + k] * B[MAX * k + x];
//
//  __syncthreads();
//
//  for (int i = 0; i < MAX; i++) {
//    sum += shmem[i];
//  }
//  C[MAX * y + x] += sum;
//}

__global__ void matrix_multiplication_with_shared_memory(double* matA, double* matB, double* matC, int m, int n, int k)
{
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  double val = 0;
  __shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];

  int localRow = threadIdx.x;
  int localCol = threadIdx.y;

  for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
    int offset = bID * BLOCK_SIZE;

    // load A and B
    if (row >= m || offset + localCol >= k)
      subA[localRow][localCol] = 0;
    else
      subA[localRow][localCol] = matA[ID2INDEX(row, offset + localCol, k)];

    if (col >= n || offset + localRow >= k)
      subB[localRow][localCol] = 0;
    else
      subB[localRow][localCol] = matB[ID2INDEX(offset + localRow, col, n)];

    __syncthreads();

    // compute
    LOOP_I(BLOCK_SIZE) {
      val += KERNEL_MUL(subA[localRow][i], subB[i][localCol]);
    }
    __syncthreads();
  }

  if (row >= m || col >= n)
    return;

  matC[ID2INDEX(row, col, n)] = val;
}


int main() {
  double A[MAX * MAX], B[MAX * MAX], C[MAX * MAX];
  float time, time_avg = 0;
  cudaEvent_t start, stop;

  cudaEventCreate(&start); // create start event
  cudaEventCreate(&stop); // create stop event
  cudaEventRecord(start, 0); // record start event

  for (int i = 0; i < MAX * MAX; i++) { // create A, B data
    A[i] = 1.0;
    B[i] = 1.0;
    C[i] = 0.0;
  }

  for (int t = 0; t < EPOCH; t++) { // EPOCH만큼 반복하여 평균시간을 계산
    double* cuda_A = 0;
    double* cuda_B = 0;
    double* cuda_C = 0;
    cudaMalloc((void**)&cuda_A, MAX * MAX * sizeof(double));
    cudaMalloc((void**)&cuda_B, MAX * MAX * sizeof(double));
    cudaMalloc((void**)&cuda_C, MAX * MAX * sizeof(double));

    cudaMemcpy(cuda_A, A, MAX * MAX * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, MAX * MAX * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(cuda_C, C, MAX * MAX * sizeof(double), cudaMemcpyHostToDevice);

    //dim3 dimGrid(MAX / BLOCKS, MAX / BLOCKS);
    //dim3 dimBlock(BLOCKS, BLOCKS);

    cudaEventCreate(&start); // 시간 기록
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 gridDim(ceil((double)MAX / BLOCK_SIZE), ceil((double)MAX / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    //matrix_multiplication_with_shared_memory << < dimGrid, dimBlock >> > (cuda_C, cuda_A, cuda_B); // matrix multiplication
    matrix_multiplication_with_shared_memory << < gridDim, blockDim >> > (cuda_C, cuda_A, cuda_B, MAX, MAX, MAX); // matrix multiplication

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C, cuda_C, MAX * MAX * sizeof(double), cudaMemcpyDeviceToHost);

    if (t == 0) {
      for (int i = 0; i < MAX * MAX; i++) {
        if ((i + 1) % MAX == 0) {
          printf("%.1lf\n", C[i]);
        }
        else {
          printf("%.1lf ", C[i]);
        }
      }
    }

    time_avg += time;

    // 메모리 해제
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
  }

  printf("time : %f\n", time_avg / (float)EPOCH); // 평균 소요 시간 출력
  return 0;
}
