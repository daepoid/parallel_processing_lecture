#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX 16
#define EPOCH 100
#pragma warning(disable:4996)

void addVector(int* C, int* A, int* B) {
  for (int i = 0; i < MAX; i++) {
    for (int j = 0; j < MAX; j++) {
      double sum = 0.0;
      for (int k = 0; k < MAX; k++) {
        sum += A[i * MAX + k] * B[k * MAX + j];
      }
      C[i * MAX + j] = (int)sum;
    }
  }
}

int main() {
  int A[MAX * MAX], B[MAX * MAX], C[MAX * MAX];
  clock_t time_avg = 0;
  clock_t start, end;

  for (int i = 0; i < MAX * MAX; i++) {
    A[i] = 1;
    B[i] = 1;
  }

  for (int t = 0; t < EPOCH; t++) { // EPOCH만큼 반복

    int* ptr_A = 0; // 비슷한 방식으로 진행하였을 때 걸리는 시간 차이를
    int* ptr_B = 0; // 보여주기 위해 유사한 방식으로 프로그램을 작성
    int* ptr_C = 0;
    ptr_A = (int*)malloc(sizeof(int) * MAX * MAX);
    ptr_B = (int*)malloc(sizeof(int) * MAX * MAX);
    ptr_C = (int*)malloc(sizeof(int) * MAX * MAX);

    memcpy(ptr_A, A, sizeof(int) * MAX * MAX); // 메모리에 복사하여 계산하는 
    memcpy(ptr_B, B, sizeof(int) * MAX * MAX); // 조건도 추가하고 측정하여 비교

    start = clock();
    for (int i = 0; i < MAX; i++) {
      addVector(ptr_C, ptr_A, ptr_B);
    }
    end = clock();

    memcpy(C, ptr_C, sizeof(int) * MAX * MAX);
    memset(ptr_C, 0, sizeof(C));

    if (t == 0) {
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
    }

    time_avg += (end - start);

    //printf("\n\n");
    free(ptr_A);
    free(ptr_B);
    free(ptr_C);
  }

  printf("time : %f\n", (float)time_avg / (float)EPOCH);
  return 0;
}
