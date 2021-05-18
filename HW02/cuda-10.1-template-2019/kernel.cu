#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
// 32 64 128
#define MAX 4
#define EPOCH 100
#define BLOCKS 8
#pragma warning(disable:4996)

__global__ void matrix_multiplication(int* C, int* A, int* B) {
  /*int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;*/

  int i = threadIdx.x;
  int j = blockIdx.x;

  for (int k = 0; k < MAX; k++) {
    C[MAX * j + i] += A[MAX * j + k] * B[MAX * k + i];
  }
}

int main() {
  int A[MAX * MAX], B[MAX * MAX], C[MAX * MAX] = { 0, };
  int cycle = MAX / 10;
  float time, time_avg = 0;
  cudaEvent_t start, stop;

  cudaEventCreate(&start); // create start event
  cudaEventCreate(&stop); // create stop event
  cudaEventRecord(start, 0); // record start event

  for (int i = 0; i < MAX * MAX; i++) { // create A, B data
    A[i] = 2;
    B[i] = 2;
  }

  for (int t = 0; t < EPOCH; t++) { // EPOCH만큼 반복하여 평균시간을 계산
    int* cuda_A = 0;
    int* cuda_B = 0;
    int* cuda_C = 0;
    cudaMalloc((void**)&cuda_A, MAX * MAX * sizeof(int));
    cudaMalloc((void**)&cuda_B, MAX * MAX * sizeof(int));
    cudaMalloc((void**)&cuda_C, MAX * MAX * sizeof(int));

    cudaMemcpy(cuda_A, A, MAX * MAX * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, MAX * MAX * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCKS, BLOCKS);
    dim3 numBlocks(MAX * MAX / threadsPerBlock.x);

    cudaEventCreate(&start); // 시간 기록
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matrix_multiplication << < numBlocks, threadsPerBlock >> > (cuda_C, cuda_A, cuda_B); // Vector Addition 계산

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C, cuda_C, MAX * MAX * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < MAX * MAX; i++) {
      if ((i + 1) % MAX == 0) {
        printf("%d\n", C[i]);
      }
      else {
        printf("%d ", C[i]);
      }
    }

    /*if (t == 0) {
      for (int i = 0; i < MAX; i++) {
        for (int j = 0; j < MAX; j++) {
          if (j == MAX - 1) {
            printf("%d\n", C[i * MAX + j]);
          }
          else {
            printf("%d ", C[i * MAX + j]);
          }
        }
      }
    }*/
    time_avg += time;
    //printf("\n\n");

    // 메모리 해제
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
  }

  printf("time : %f\n", time_avg / (float)EPOCH); // 평균 소요 시간 출력
  return 0;
}
