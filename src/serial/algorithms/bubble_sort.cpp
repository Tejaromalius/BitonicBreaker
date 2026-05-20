#include "../algorithms.h"

void bubble_sort(int* a, int n) {
  for (int i = 0; i < n - 1; i++) {
    bool s = 0;
    for (int j = 0; j < n - i - 1; j++)
      if (a[j] > a[j + 1]) std::swap(a[j], a[j + 1]), s = 1;
    if (!s) break;
  }
}
