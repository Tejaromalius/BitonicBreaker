#include "../algorithms.h"
#include "../utils.h"

double parallel_quick_sort(int r, int s, int* g, int curN) {
  int ln = curN / s;
  int* l = (int*)malloc(ln * sizeof(int));
  MPI_Scatter(g, ln, MPI_INT, l, ln, MPI_INT, 0, MPI_COMM_WORLD);
  double st = MPI_Wtime();
  qsort(l, ln, sizeof(int), cmp);
  int p = !r ? l[ln / 2] : 0;
  MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int it = upper_bound(l, ln, p);
  int lp_size = it;
  int hp_size = ln - it;
  int* lp = l;
  int* hp = l + it;

  int pr = r ^ (s / 2);
  int sc = r < s / 2 ? hp_size : lp_size;
  int rc = 0;
  MPI_Sendrecv(&sc, 1, MPI_INT, pr, 0, &rc, 1, MPI_INT, pr, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  int* rb = (int*)malloc(rc * sizeof(int));
  MPI_Sendrecv(r < s / 2 ? hp : lp, sc, MPI_INT, pr, 1, rb, rc, MPI_INT, pr, 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  int current_size = (r < s / 2 ? lp_size : hp_size);
  int* current_arr = (r < s / 2 ? lp : hp);
  int* new_l = mrg(current_arr, current_size, rb, rc);
  free(rb);

  double en = MPI_Wtime();
  int ls = current_size + rc;
  int* rcs = (int*)malloc(s * sizeof(int));
  int* dsp = (int*)calloc(s, sizeof(int));
  MPI_Gather(&ls, 1, MPI_INT, rcs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!r) {
    for (int i = 1; i < s; i++) dsp[i] = dsp[i - 1] + rcs[i - 1];
  }
  MPI_Gatherv(new_l, ls, MPI_INT, g, rcs, dsp, MPI_INT, 0, MPI_COMM_WORLD);

  free(l);
  free(new_l);
  free(rcs);
  free(dsp);
  return en - st;
}
