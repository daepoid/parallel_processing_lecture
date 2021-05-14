#include <cstring>
#include <iostream>
#include <thread>
#include <windows.h>
#define ROW 6
#define COL 8
using namespace std;

bool isvalid(int i, int j) {
  return 0 < i && i < ROW + 1 && 0 < j && j < COL + 1;
}

void init(double board[][COL + 2], int thread_board[][COL + 2]) {
  int temp[ROW + 2][COL + 2] = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 1, 2, 2, 3, 3, 3, 0},
      {0, 1, 1, 1, 2, 2, 3, 3, 3, 0}, {0, 4, 4, 2, 5, 5, 6, 6, 6, 0},
      {0, 4, 4, 4, 5, 5, 8, 6, 6, 0}, {0, 7, 7, 7, 8, 8, 9, 9, 9, 0},
      {0, 7, 7, 7, 8, 8, 9, 9, 9, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };
  memcpy(thread_board, temp, sizeof(int) * (ROW + 2) * (COL + 2));

  for (int i = 0; i < ROW + 2; i++) {
    for (int j = 0; j < COL + 2; j++) {
      if (isvalid(i, j)) {
        board[i][j] = 100;
      } else {
        board[i][j] = 0;
      }
    }
  }
}

void print_board(double board[][COL + 2]) {
  for (int i = 0; i < ROW + 2; i++) {
    for (int j = 0; j < COL + 2; j++) {
      if (j == COL + 1) {
        printf("%lf\n", board[i][j]);
      } else {
        printf("%lf ", board[i][j]);
      }
    }
  }
}

void calculate(double board[][COL + 2], double new_board[][COL + 2],
               int thread_board[][COL + 2], int thread_num) {
  for (int i = 1; i < ROW + 1; i++) {
    for (int j = 1; j < COL + 1; j++) {
      if (thread_board[i][j] == thread_num) {
        new_board[i][j] = board[i][j - 1] + board[i][j + 1] + board[i - 1][j] +
                          board[i + 1][j];
        new_board[i][j] /= 4;
      }
    }
  }
}

void use_multi_thread(double board[][COL + 2], int thread_board[][COL + 2]) {
  double new_board[ROW + 2][COL + 2];
  for (int i = 0; i < ROW + 2; i++) {
    memset(new_board[i], 0, sizeof(double) * (COL + 2));
  }

  //쓰레드 별로 수행
  thread t1(calculate, board, new_board, thread_board, 1);
  thread t2(calculate, board, new_board, thread_board, 2);
  thread t3(calculate, board, new_board, thread_board, 3);
  thread t4(calculate, board, new_board, thread_board, 4);
  thread t5(calculate, board, new_board, thread_board, 5);
  thread t6(calculate, board, new_board, thread_board, 6);
  thread t7(calculate, board, new_board, thread_board, 7);
  thread t8(calculate, board, new_board, thread_board, 8);
  thread t9(calculate, board, new_board, thread_board, 9);

  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
  t6.join();
  t7.join();
  t8.join();
  t9.join();

  //원래 board에 값을 갱신
  memcpy(board, new_board, sizeof(double) * (ROW + 2) * (COL + 2));
}

int main() {
  int thread_board[ROW + 2][COL + 2];
  double board[ROW + 2][COL + 2];

  // 초기화
  init(board, thread_board);
  // 초기상태 출력
  print_board(board);

  for (int i = 1; i <= 100; i++) {
    use_multi_thread(board, thread_board);
    if (i % 5 == 0) {
      system("cls");
      printf("\n\n%.1lf초 경과\n", i * 0.1);
      print_board(board);
      Sleep(1000);
    }
  }
}
