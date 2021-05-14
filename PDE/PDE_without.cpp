//#include <process.h>
//#include <windows.h>
//
//#include <cstdlib>
//#include <cstring>
//#include <iostream>
//using namespace std;
//
//#define ROW 6
//#define COL 8
//
//int isvalid(int i, int j) {
//  return 0 < i && i < ROW + 1 && 0 < j && j < COL + 1;
//}
//
//void init(double board[][COL + 2]) {
//  for (int i = 0; i < ROW + 2; i++) {
//    for (int j = 0; j < COL + 2; j++) {
//      if (isvalid(i, j) == 1) {
//        board[i][j] = 100;
//      } else {
//        board[i][j] = 0;
//      }
//    }
//  }
//}
//
//void print_board(double board[][COL + 2]) {
//  for (int i = 0; i < ROW + 2; i++) {
//    for (int j = 0; j < COL + 2; j++) {
//      if (j == COL + 1) {
//        printf("%lf", board[i][j]);
//      } else {
//        printf("%lf ", board[i][j]);
//      }
//    }
//    printf("\n");
//  }
//}
//
//void use_one_thread(double board[][COL + 2]) {
//  double new_board[ROW + 2][COL + 2];
//  for (int i = 0; i < ROW + 2; i++) {
//    memset(new_board[i], 0, sizeof(double) * (COL + 2));
//  }
//  for (int i = 1; i < ROW + 1; i++) {
//    for (int j = 1; j < COL + 1; j++) {
//      new_board[i][j] =
//          board[i][j - 1] + board[i][j + 1] + board[i - 1][j] + board[i + 1][j];
//      new_board[i][j] /= 4;
//    }
//  }
//  memcpy(board, new_board, sizeof(double) * (ROW + 2) * (COL + 2));
//}
//
//void use_multi_thread() {}
//
//int main() {
//  double board[ROW + 2][COL + 2];
//
//  init(board);
//  print_board(board);
//
//  for (int i = 0; i < 100; i++) {
//    use_one_thread(board);
//    if (i % 10 == 0) {
//      printf("%d\n", i);
//      print_board(board);
//    }
//  }
//  printf("\n");
//  print_board(board);
//}
