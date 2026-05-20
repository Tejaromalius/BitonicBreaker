#ifndef SERIAL_ALGORITHMS_H
#define SERIAL_ALGORITHMS_H

#include <algorithm>
#include <climits>
#include <vector>

void bubble_sort(int* a, int n);
void selection_sort(int* a, int n);
void quick_sort(int* a, int l, int h);
void merge_sort(int* a, int l, int r);
void bitonic_sort(int* a, int n);

#endif
