#include "../algorithms.h"

void selection_sort(int* a, int n) {
  for (int i = 0; i < n - 1; i++) {
    int m = i;
    for (int j = i + 1; j < n; j++)
      if (a[j] < a[m]) m = j;
    if (m != i) std::swap(a[i], a[m]);
  }
}
