#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

double a[4] = {1, 2, 3, 4};
double b = 0;

void* f1(void* data) {
  pthread_mutex_t* mutex = (pthread_mutex_t *) data;
  double t = a[0] + a[1];
  pthread_mutex_lock(mutex);
  double z = b;
  //usleep(1); // sleep for 1 microsecond
  b = z + t;
  pthread_mutex_unlock(mutex);

  return NULL;
}

void* f2(void* data) {
  pthread_mutex_t* mutex = (pthread_mutex_t*) data;
  double t = a[2] + a[3];
  pthread_mutex_lock(mutex);
  b = b + t;
  pthread_mutex_unlock(mutex);

  return NULL;
}

int main() {
  pthread_t t1, t2;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;


  pthread_create(&t1, NULL, f1, (void *)&mutex);
  pthread_create(&t2, NULL, f2, (void *)&mutex);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  printf("%f\n", b);

  return 0;
}
