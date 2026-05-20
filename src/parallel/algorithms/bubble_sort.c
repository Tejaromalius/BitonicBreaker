#include "../algorithms.h"
#include "../utils.h"

double parallel_bubble_sort(int r, int s, int* g, int curN) {
  int ln = curN / s;
  int* l = (int*)malloc(ln * sizeof(int));
  MPI_Scatter(g, ln, MPI_INT, l, ln, MPI_INT, 0, MPI_COMM_WORLD);
  double st = MPI_Wtime();
  qsort(l, ln, sizeof(int), cmp);
  for (int p = 0; p < s; p++) {
    int pr = (p % 2) ? (r % 2 ? r - 1 : r + 1) : (r % 2 == 0 ? r + 1 : r - 1);
    if (pr >= 0 && pr < s) {
      int* rv = (int*)malloc(ln * sizeof(int));
      MPI_Sendrecv(l, ln, MPI_INT, pr, 0, rv, ln, MPI_INT, pr, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int* m = mrg(l, ln, rv, ln);
      memcpy(l, m + (r < pr ? 0 : ln), ln * sizeof(int));
      free(m);
      free(rv);
    }
  }
  double en = MPI_Wtime();
  MPI_Gather(l, ln, MPI_INT, g, ln, MPI_INT, 0, MPI_COMM_WORLD);
  free(l);
  return en - st;
}
