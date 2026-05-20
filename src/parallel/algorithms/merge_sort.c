#include "../algorithms.h"
#include "../utils.h"

double parallel_merge_sort(int r, int s, int* g, int curN) {
  int ln = curN / s;
  int* l = (int*)malloc(ln * sizeof(int));
  MPI_Scatter(g, ln, MPI_INT, l, ln, MPI_INT, 0, MPI_COMM_WORLD);
  double st = MPI_Wtime();
  qsort(l, ln, sizeof(int), cmp);

  int curr_size = ln;
  for (int i = 1; i < s; i *= 2) {
    if (r % (2 * i) == 0 && r + i < s) {
      int rn;
      MPI_Recv(&rn, 1, MPI_INT, r + i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int* rb = (int*)malloc(rn * sizeof(int));
      MPI_Recv(rb, rn, MPI_INT, r + i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int* new_l = mrg(l, curr_size, rb, rn);
      free(l);
      free(rb);
      l = new_l;
      curr_size += rn;
    } else if (r % i == 0) {
      MPI_Send(&curr_size, 1, MPI_INT, r - i, 0, MPI_COMM_WORLD);
      MPI_Send(l, curr_size, MPI_INT, r - i, 1, MPI_COMM_WORLD);
      break;
    }
  }
  if (!r) memcpy(g, l, curr_size * sizeof(int));
  free(l);
  return MPI_Wtime() - st;
}
