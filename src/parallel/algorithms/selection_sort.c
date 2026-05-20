#include "../algorithms.h"
#include "../utils.h"

double parallel_selection_sort(int r, int s, int* g, int curN) {
  int ln = curN / s;
  int* l = (int*)malloc(ln * sizeof(int));
  MPI_Scatter(g, ln, MPI_INT, l, ln, MPI_INT, 0, MPI_COMM_WORLD);
  double st = MPI_Wtime();
  qsort(l, ln, sizeof(int), cmp);
  MPI_Gather(l, ln, MPI_INT, g, ln, MPI_INT, 0, MPI_COMM_WORLD);
  if (!r) {
    int* p = (int*)calloc(s, sizeof(int));
    int* res = (int*)malloc(curN * sizeof(int));
    for (int c = 0; c < curN; c++) {
      int mv = INT_MAX, ms = -1;
      for (int i = 0; i < s; i++) {
        if (p[i] < ln && g[i * ln + p[i]] < mv) {
          mv = g[i * ln + p[i]];
          ms = i;
        }
      }
      res[c] = mv;
      p[ms]++;
    }
    memcpy(g, res, curN * sizeof(int));
    free(p);
    free(res);
  }
  double en = MPI_Wtime();
  free(l);
  return en - st;
}
