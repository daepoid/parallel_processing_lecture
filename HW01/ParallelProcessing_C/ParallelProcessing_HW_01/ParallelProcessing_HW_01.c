#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX 16384
#define EPOCH 100
#pragma warning(disable:4996)

void addVector(int* C, int* A, int* B) {
  static int idx = 0;
  C[idx % MAX] = A[idx % MAX] + B[idx % MAX];
  idx = (idx + 1) % MAX;
}

int main() {
  int A[MAX], B[MAX], C[MAX];
  int cycle = MAX / 10;
  clock_t time_avg = 0;
  clock_t start, end;

  for (int i = 0; i < MAX; i++) {
    A[i] = i;
    B[i] = i * i;
  }

  for (int t = 0; t < EPOCH; t++) { // EPOCH��ŭ �ݺ�

    int* ptr_A = 0; // ����� ������� �����Ͽ��� �� �ɸ��� �ð� ���̸�
    int* ptr_B = 0; // �����ֱ� ���� ������ ������� ���α׷��� �ۼ�
    int* ptr_C = 0;
    ptr_A = (int*)malloc(sizeof(int) * MAX);
    ptr_B = (int*)malloc(sizeof(int) * MAX);
    ptr_C = (int*)malloc(sizeof(int) * MAX);

    memcpy(ptr_A, A, sizeof(int) * MAX); // �޸𸮿� �����Ͽ� ����ϴ� 
    memcpy(ptr_B, B, sizeof(int) * MAX); // ���ǵ� �߰��ϰ� �����Ͽ� ��

    start = clock();
    for (int i = 0; i < MAX; i++) {
      addVector(ptr_C, ptr_A, ptr_B);
    }
    end = clock();

    memcpy(C, ptr_C, sizeof(int) * MAX);
    memset(ptr_C, 0, sizeof(C));

    for (int i = 0; i < MAX; i++) {
      if (i % cycle == 0) {
        printf("%d + %d = %d\n", A[i], B[i], C[i]);
      }
    }

    time_avg += (end - start);

    printf("\n\n");
    free(ptr_A);
    free(ptr_B);
    free(ptr_C);
  }

  printf("time : %f\n", (float)time_avg / (float)EPOCH);
  return 0;
}
