#include <time.h>

inline int64_t get_ts() 
{
  timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return static_cast<int64_t>(tp.tv_sec) * 1000 * 1000 * 1000 + static_cast<int64_t>(tp.tv_nsec);
}

