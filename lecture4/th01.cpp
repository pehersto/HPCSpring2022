#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

double a[4] = {1, 2, 3, 4};
double b = 0;

void* f1(void* in) {
  double t = a[0] + a[1];
  double z = b;
  //usleep(1); // sleep for 1 microsecond
  b = z + t;

  return NULL;
}

void* f2(void* in) {
  double t = a[2] + a[3];
  b = b + t;

  return NULL;
}

int main() {
  pthread_t t1, t2;

  pthread_create(&t1, NULL, f1, NULL);
  pthread_create(&t2, NULL, f2, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  printf("%f\n", b);

  return 0;
}
