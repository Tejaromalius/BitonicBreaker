#include "../algorithms.h"

void merge_sort(int* a, int l, int r) {
  if (l >= r) return;
  int d = l + (r - l) / 2;
  merge_sort(a, l, d);
  merge_sort(a, d + 1, r);
  std::inplace_merge(a + l, a + d + 1, a + r + 1);
}
