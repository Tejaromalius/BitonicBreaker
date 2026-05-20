#ifndef PARALLEL_UTILS_H
#define PARALLEL_UTILS_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

int cmp(const void* a, const void* b);
int* mrg(const int* a, int sa, const int* b, int sb);
int upper_bound(int* arr, int len, int val);
int is_sorted(int* a, int n);

#endif
