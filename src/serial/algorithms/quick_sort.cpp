#include "../algorithms.h"

void quick_sort(int* a, int l, int h) {
  if (l >= h) return;
  int m = l + (h - l) / 2;
  if (a[m] < a[l]) std::swap(a[m], a[l]);
  if (a[h] < a[l]) std::swap(a[h], a[l]);
  if (a[m] < a[h]) std::swap(a[m], a[h]);
  int p = a[h], i = l - 1;
  for (int j = l; j < h; j++)
    if (a[j] <= p) std::swap(a[++i], a[j]);
  std::swap(a[i + 1], a[h]);
  quick_sort(a, l, i);
  quick_sort(a, i + 2, h);
}
