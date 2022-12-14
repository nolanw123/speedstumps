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
// the instrinsics corresponding to the "basic idea" above are:
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX&expand=486,518,848,848&text=_mm_cmp_ps
//
// and
//
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE4_1&expand=486,518&text=_mm_blendv_ps
//
// handy note:
//
// we can get an annotated dump like this:
// objdump -d -M intel -S objs/opt/vectest.o  > vectest.asm
//

#include <immintrin.h>

#include <iostream>
#include <random>

#include "util.h"

inline float horizontal_add(__m256 &a) {
  a = _mm256_hadd_ps(a,a);
  a = _mm256_hadd_ps(a,a);
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

int main(int argc, char **argv)
{
  const size_t TRIALS = 200;
  const size_t COUNT = 8*100000;

  __m256 *a = new __m256[COUNT/8];
  __m256 *b = new __m256[COUNT/8];
  __m256 *x = new __m256[COUNT/8];
  __m256 *y = new __m256[COUNT/8];

  std::mt19937_64 g(1234);      // mersenne twister with constant seed for reproducibility
  std::uniform_real_distribution<float> d(-0.1, 0.1);
  for(size_t i = 0 ; i < COUNT/8 ; ++i) {
    a[i] = _mm256_set_ps(d(g), d(g), d(g), d(g), d(g), d(g), d(g), d(g));
    b[i] = _mm256_set_ps(d(g), d(g), d(g), d(g), d(g), d(g), d(g), d(g));
    x[i] = _mm256_set_ps(d(g), d(g), d(g), d(g), d(g), d(g), d(g), d(g));
    y[i] = _mm256_set_ps(d(g), d(g), d(g), d(g), d(g), d(g), d(g), d(g));
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

  return 0;
}
