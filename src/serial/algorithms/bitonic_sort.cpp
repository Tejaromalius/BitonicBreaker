#include "../algorithms.h"

void bm(int* a, int l, int c, bool s) {
  if (c < 2) return;
  int k = c / 2;
  for (int i = l; i < l + k; i++)
    if ((a[i] > a[i + k]) == s) std::swap(a[i], a[i + k]);
  bm(a, l, k, s);
  bm(a, l + k, k, s);
}

void bs(int* a, int l, int c, bool s) {
  if (c < 2) return;
  int k = c / 2;
  bs(a, l, k, 1);
  bs(a, l + k, k, 0);
  bm(a, l, c, s);
}

void bitonic_sort(int* a, int n) {
  int p = 1;
  while (p < n) p <<= 1;
  std::vector<int> b(p, INT_MAX);
  std::copy(a, a + n, b.begin());
  bs(b.data(), 0, p, 1);
  std::copy(b.begin(), b.begin() + n, a);
}
