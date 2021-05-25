#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX 16
#define EPOCH 100
#pragma warning(disable:4996)

void matrix_multiplication(double* C, double* A, double* B, int i, int j) {
  double sum = 0.0;
  for (int k = 0; k < MAX; k++) {
    sum += A[MAX * i + k] * B[MAX * k + j];
  }
  C[MAX * i + j] = sum;
}

int main() {
  double A[MAX * MAX], B[MAX * MAX], C[MAX * MAX];
  clock_t time_avg = 0;
  clock_t start, end;

  for (int i = 0; i < MAX * MAX; i++) {
    A[i] = 1;
    B[i] = 1;
    C[i] = 0;
  }

  for (int t = 0; t < EPOCH; t++) { // EPOCH��ŭ �ݺ�

    double* ptr_A = (double*)malloc(sizeof(double) * MAX * MAX); // ����� ������� �����Ͽ��� �� �ɸ��� �ð� ���̸�
    double* ptr_B = (double*)malloc(sizeof(double) * MAX * MAX); // �����ֱ� ���� ������ ������� ���α׷��� �ۼ�
    double* ptr_C = (double*)malloc(sizeof(double) * MAX * MAX);

    memcpy(ptr_A, A, sizeof(double) * MAX * MAX); // �޸𸮿� �����Ͽ� ����ϴ� 
    memcpy(ptr_B, B, sizeof(double) * MAX * MAX); // ���ǵ� �߰��ϰ� �����Ͽ� ��

    start = clock();
    for (int i = 0; i < MAX; i++) {
      for (int j = 0; j < MAX; j++) {
        matrix_multiplication(ptr_C, ptr_A, ptr_B, i, j);
      }
    }
    end = clock();
    time_avg += (double)(end - start);

    memcpy(C, ptr_C, sizeof(double) * MAX * MAX);

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

    free(ptr_A);
    free(ptr_B);
    free(ptr_C);
  }

  printf("time : %lf\n", (double)time_avg / (double)EPOCH);
  return 0;
}
