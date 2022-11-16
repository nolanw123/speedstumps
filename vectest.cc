//
// a test program for decision stumps
//
// basic idea:
//
// 1) do a simd comparison of values to generate a mask
// 2) use the mask to select values
//
// i.e. if we had some logic like:
//
// if(a[i] <= b[i]) {
//   tot += x[i];
// } else {
//   tot += y[i];
// }
//
// we could do this in parallel with simd instructions, and avoid branching entirely
//
//
// the instrinsics corresponding to the "basic idea" above are:
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX&expand=486,518,848,848&text=_mm_cmp_ps
//
// and
//
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE4_1&expand=486,518&text=_mm_blendv_ps
//
// handy note:
// we can get an annotated dump like this:
// objdump -d -M intel -S objs/opt/vectest.o  > vectest.asm
//

#include <stdint.h>
#include <immintrin.h>
#include <xmmintrin.h>

#include <iostream>
#include <random>

#include "util.h"

inline float horizontal_add(__m256 &a) {
  a = _mm256_hadd_ps(a,a);
  a = _mm256_hadd_ps(a, a);
  __m128 t1 = _mm256_extractf128_ps(a,1);
  t1 = _mm_add_ss(_mm256_castps256_ps128(a),t1);
  return _mm_cvtss_f32(t1);
}

inline float horizontal_add(__m128 &a) {
  a = _mm_hadd_ps(a,a);
  a = _mm_hadd_ps(a,a);
  return _mm_cvtss_f32(a);
}

// 256-bit simd implementation
float selectf(float *a, float *b, float *x, float *y, size_t count)
{
  __m256 *ap = (__m256*)a;
  __m256 *bp = (__m256*)b;
  __m256 *xp = (__m256*)x;
  __m256 *yp = (__m256*)y;  
  __m256 tot = _mm256_setzero_ps();

  for(size_t i = 0 ; i < (count >> 3) ; ++i) {
    __m256 mask = _mm256_cmp_ps(*ap++, *bp++, 30); // _CMP_GT_OQ aka > (ie the OPPOSITE of <= because we want an inverse result in the mask)
    __m256 res = _mm256_blendv_ps(*xp++, *yp++, mask);
    tot = _mm256_add_ps(tot, res); // vertically accumulate results
  }
  
  return horizontal_add(tot) / count;
}

// 128-bit simd implementation
float selectf2(float *a, float *b, float *x, float *y, size_t count)
{
  __m128 *ap = (__m128*)a;
  __m128 *bp = (__m128*)b;
  __m128 *xp = (__m128*)x;
  __m128 *yp = (__m128*)y;  
  __m128 tot = _mm_setzero_ps();

  for(size_t i = 0 ; i < (count >> 2) ; ++i) {
    __m128 mask = _mm_cmp_ps(*ap++, *bp++, 30); // _CMP_GT_OQ aka > (ie the OPPOSITE of <= because we want an inverse result in the mask)
    __m128 res = _mm_blendv_ps(*xp++, *yp++, mask);
    tot = _mm_add_ps(tot, res); // vertically accumulate results
  }
  
  return horizontal_add(tot) / count;
  
}

// this is the traditional (slow) decision stump evaluation function 
float selectslow(float *a, float *b, float *x, float *y, size_t count)
{
  float total = 0.0;
  for(size_t i = 0 ; i < count ; ++i) {
    if(a[i] <= b[i]) {
      total += x[i];
    } else {
      total += y[i];
    }
  }
  return total / count;
}

// 16-bit aligned a/b, mask, and results array, do selection for count (which must be a multiple of 4)
// this version uses the idea of keeping the arguments to _mm_blendv_ps next to each other in groups
// -- some experiments with cachegrind showed that keeping the result in the struct (which makes
// the struct 64 bytes, the size of a cache line) result in maybe 1/4 missed cache writes (why?)
// -- keeping the results in their own array was faster.  With this we get results that are
// roughly 70% of the selectf2 numbers.  48/64 is 75% which is interesting.
#pragma pack(push, 1)
struct selectf2_2_entry {
  __m128 a,b,mask;
};
static_assert(sizeof(selectf2_2_entry) == 48, "Size of selectf2_2_entry is not 48 bytes");
#pragma pack(pop)

float selectf2_2(selectf2_2_entry *entries, size_t count)
{
  __m128 tot;
  
  for(size_t i = 0 ; i < (count >> 2) ; ++i) {
    __m128 res = _mm_blendv_ps(entries->a, entries->b, entries->mask);
    tot = _mm_add_ps(tot, res); // vertically accumulate results    
    ++entries;
  }

  tot = _mm_hadd_ps(tot, tot); // now horizontally accumulate results
  tot = _mm_hadd_ps(tot, tot); // need to do 2x because hadd_ps works in pairs

  return VGBI<float, 0>(tot) / count;  
}

int main(int argc, char **argv)
{
  const size_t TRIALS = 200;
  
  const size_t COUNT = 8*100000;

  __m256 *mask = new __m256[COUNT/8]; //{ 0xffffffff,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0xffffffff};
  mask[0][0] = 0xffffffff;
  mask[5][0] = 0xffffffff;  

  if(argv[1][0] == 'a') {
    mask[1][0] = 0xffffffff;
  }

  __m256 *a = new __m256[COUNT/8];
  __m256 *b = new __m256[COUNT/8];  
  __m256 *x = new __m256[COUNT/8];
  __m256 *y = new __m256[COUNT/8];
  selectf2_2_entry *entries2 = new selectf2_2_entry[COUNT/4]; // /4 because using __m128 internally

  std::mt19937_64 g(1234);      // mersenne twister with constant seed for reproducibility
  std::uniform_real_distribution<float> d(-0.1, 0.1);
  for(size_t i = 0 ; i < COUNT/8 ; ++i) {
    {
      float vals[8] = { d(g), d(g), d(g), d(g), d(g), d(g), d(g), d(g) };
      a[i] = _mm256_set_ps(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
    }
    {
      float vals[8] = { d(g), d(g), d(g), d(g), d(g), d(g), d(g), d(g) };
      b[i] = _mm256_set_ps(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
    }
    {
      float vals[8] = { d(g), d(g), d(g), d(g), d(g), d(g), d(g), d(g) };
      x[i] = _mm256_set_ps(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
    }
    {
      float vals[8] = { d(g), d(g), d(g), d(g), d(g), d(g), d(g), d(g) };
      y[i] = _mm256_set_ps(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
    }    
  }
    
  std::cout << "Running tests on " << COUNT << " elements" << std::endl;

  auto timer = []<typename FUNC>(FUNC f_, size_t trials_, const std::string &name_) {
    int64_t start,end;
    double total = 0.0;
    double val = 0.0;
    for(size_t trial = 0 ; trial < trials_ ; ++trial) {
      start = get_ts();
      val = f_();
      end = get_ts();
      total += (end - start);
    }
    total /= trials_;
    std::cout << (uint64_t)total << " nanos/trial (" << trials_ << " trials) for " << name_ << " (val=" << val << ")" << std::endl;
  };

  timer([&](){ return selectslow(&a[0][0],&b[0][0],&x[0][0],&y[0][0],COUNT); }, TRIALS, "selectslow");
  timer([&](){ return selectf(&a[0][0],&b[0][0],&x[0][0],&y[0][0],COUNT); }, TRIALS, "selectf");
  timer([&](){ return selectf2(&a[0][0],&b[0][0],&x[0][0],&y[0][0],COUNT); }, TRIALS, "selectf2");
  timer([&](){ return selectf2_2(&entries2[0],COUNT); }, TRIALS, "selectf2_2");

  return 0;
}
