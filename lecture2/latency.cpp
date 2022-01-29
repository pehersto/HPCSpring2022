// Measuring memory latency for different array sizes
// $ g++ -std=c++11 -O3 -march=native -g latency.cpp && ./a.out
// $ getconf -a | grep CACHE

// Compare sequential vs random access patterns
#define STRIDE 1

#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

constexpr long cacheline = 64; // in bytes

struct SortPair {
  double key;
  long data;
  int operator<(const SortPair& p1) const { return key < p1.key; }
};

void create_random_indices(long* arr, long N) {
  constexpr long skip = cacheline / sizeof(long);
#if STRIDE
  for (long i = 0; i < N; i++) {
    arr[i * skip] = ((i+STRIDE)%N) * skip;
  }
#else
  std::vector<SortPair> v(N);
  for (long i = 0; i < N; i++) {
    v[i].key = drand48();
    v[i].data = i;
  }
  std::sort(v.begin(), v.end());
  for (long i = 0; i < N; i++) {
    arr[v[i].data * skip] = v[(i+1)%N].data * skip;
  }
#endif
}

long measure_latency(long memory_size) {
  long repeat = 1e10 / memory_size;
  long N = memory_size / cacheline;
  long* arr = (long*) malloc(memory_size);
  create_random_indices(arr, N);

  Timer t;
  t.tic();
  long idx = 0;
  for (long k = 0; k < repeat * N; k++) {
    idx = arr[idx];
  }
  printf("%10ld  %10f %10f\n", memory_size, t.toc()*3.3e9/N/repeat, t.toc()/N/repeat/1e-9);

  return idx;
}

int main(int argc, char** argv) {
  long max_size = 1L*1024*1024*1024;
  printf("     bytes cycles/read    ns/read\n");
  for (long i = cacheline; i < max_size; i*=2) measure_latency(i);
  return 0;
}
