#include "utils.h"

double t() {
  timeval v;
  gettimeofday(&v, 0);
  return v.tv_sec + v.tv_usec * 1e-6;
}
