#include <thread>
#include <mutex>
#include <unistd.h>

double a[4] = {1, 2, 3, 4};
double b = 0;

void f1(std::mutex* mutex) {
  double t = a[0] + a[1];
  mutex->lock();
  double z = b;
  //usleep(1); // sleep for 1 microsecond
  b = z + t;
  mutex->unlock();
}

void f2(std::mutex* mutex) {
  double t = a[2] + a[3];
  mutex->lock();
  b = b + t;
  mutex->unlock();
}

int main() {
  std::mutex mutex;
  std::thread t1(f1, &mutex);
  std::thread t2(f2, &mutex);

  t1.join();
  t2.join();

  printf("%f\n", b);

  return 0;
}
