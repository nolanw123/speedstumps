#include <time.h>

inline int64_t get_ts() 
{
  timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return static_cast<int64_t>(tp.tv_sec) * 1000 * 1000 * 1000 + static_cast<int64_t>(tp.tv_nsec);
}

// treating IT as an array of type T, return get the I'th element
//
// i.e.:
//
// __m128i foo;
// return VGBI<uint32_t, 3>(foo); // get 4th uint32_t from foo
//
template<typename T, size_t I, typename IT>
T VGBI(IT V) {
  union {
    IT v;
    T a[4];
  } converter;
  converter.v = V;
  return converter.a[I];
}
