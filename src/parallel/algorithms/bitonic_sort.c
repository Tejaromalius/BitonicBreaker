#include "../algorithms.h"
#include "../utils.h"

double parallel_bitonic_sort(int r, int s, int* g, int curN) {
  // Determine power of 2 for padding
  int p_size = 1;
  while (p_size < curN) p_size <<= 1;

  int* padded_g = NULL;
  int padded_N = p_size;
  int ln = padded_N / s;

  if (!r) {
    padded_g = (int*)malloc(padded_N * sizeof(int));
    memcpy(padded_g, g, curN * sizeof(int));
    for (int i = curN; i < padded_N; i++) padded_g[i] = INT_MAX;
  }

  int* l = (int*)malloc(ln * sizeof(int));
  MPI_Scatter(padded_g, ln, MPI_INT, l, ln, MPI_INT, 0, MPI_COMM_WORLD);

  double st = MPI_Wtime();
  qsort(l, ln, sizeof(int), cmp);

  int lg = 0;
  for (int t = s; t > 1; t >>= 1) lg++;

  for (int i = 0; i < lg; i++) {
    for (int j = i; j >= 0; j--) {
      int pr = r ^ (1 << j);
      int as = !((r >> (i + 1)) & 1);
      int* rv = (int*)malloc(ln * sizeof(int));
      MPI_Sendrecv(l, ln, MPI_INT, pr, 0, rv, ln, MPI_INT, pr, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int* mg = mrg(l, ln, rv, ln);
      int keep_left = ((r < pr) == as);
      memcpy(l, mg + (keep_left ? 0 : ln), ln * sizeof(int));
      free(mg);
      free(rv);
    }
  }

  double en = MPI_Wtime();
  MPI_Gather(l, ln, MPI_INT, padded_g, ln, MPI_INT, 0, MPI_COMM_WORLD);

  if (!r) {
    memcpy(g, padded_g, curN * sizeof(int));  // Extract original size
    free(padded_g);
  }
  free(l);
  return en - st;
}
