#pragma comment(lib,"cuda.lib")
#pragma comment(lib,"cudart.lib")

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
// 1024 4096 16384
#define MAX 16384
#define EPOCH 100
#define BLOCKS 8
#pragma warning(disable:4996)

__global__ void addVector(int* C, int* A, int* B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}

int main() {
  int A[MAX], B[MAX], C[MAX];
  int cycle = MAX / 10;
  float time, time_avg = 0;
  cudaEvent_t start, stop;

  cudaEventCreate(&start); // create start event
  cudaEventCreate(&stop); // create stop event
  cudaEventRecord(start, 0); // record start event

  for (int i = 0; i < MAX; i++) { // create A, B data
    A[i] = i;
    B[i] = i * i;
  }

  for (int t = 0; t < EPOCH; t++) { // EPOCH만큼 반복하여 평균시간을 계산
    int* cuda_A = 0;
    int* cuda_B = 0;
    int* cuda_C = 0;
    cudaMalloc((void**)&cuda_A, MAX * sizeof(int));
    cudaMalloc((void**)&cuda_B, MAX * sizeof(int));
    cudaMalloc((void**)&cuda_C, MAX * sizeof(int));

    cudaMemcpy(cuda_A, A, MAX * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, MAX * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCKS, BLOCKS);
    dim3 numBlocks(MAX / threadsPerBlock.x);

    cudaEventCreate(&start); // 시간 기록
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    addVector << < numBlocks, threadsPerBlock >> > (cuda_C, cuda_A, cuda_B); // Vector Addition 계산

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C, cuda_C, MAX * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < MAX; i++) {
      if (i % cycle == 0) {
        printf("%d + %d = %d\n", A[i], B[i], C[i]);
      }
    }

    time_avg += time;
    printf("\n\n");
  }

  printf("time : %f\n", time_avg / (float)EPOCH); // 평균 소요 시간 출력
  return 0;
}
