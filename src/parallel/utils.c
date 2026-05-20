#include "utils.h"

int cmp(const void* a, const void* b) {
  int arg1 = *(const int*)a;
  int arg2 = *(const int*)b;
  if (arg1 < arg2) return -1;
  if (arg1 > arg2) return 1;
  return 0;
}

int* mrg(const int* a, int sa, const int* b, int sb) {
  int* r = (int*)malloc((sa + sb) * sizeof(int));
  int i = 0, j = 0, k = 0;
  while (i < sa && j < sb) {
    if (a[i] <= b[j])
      r[k++] = a[i++];
    else
      r[k++] = b[j++];
  }
  while (i < sa) r[k++] = a[i++];
  while (j < sb) r[k++] = b[j++];
  return r;
}

int upper_bound(int* arr, int len, int val) {
  int left = 0, right = len;
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] <= val)
      left = mid + 1;
    else
      right = mid;
  }
  return left;
}

int is_sorted(int* a, int n) {
  for (int i = 0; i < n - 1; i++) {
    if (a[i] > a[i + 1]) return 0;
  }
  return 1;
}
