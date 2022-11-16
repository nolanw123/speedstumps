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
// if(x[i] >= y[i]) {
//   v[i] = a[i];
// } else {
//   v[i] = b[i];
// }
//
// we could do this in parallel with simd instructions, and avoid branching entirely
//
// note: it may be the case we can speed things up even more by arranging the
// layouts of x,y,a,b, and v in memory properly
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
// some todo's:
// - selectslow might be faster if we re-arranged the memory layout
//   so that a,b,x,y,and res were all next to each other in chunks
//   ie: a0,b0,x0,y0,res0,a1,b1,x1,y1,res1, etc
//   then we'd just be striding through memory and everything would be in cache
// - the same thing might apply to the selectf(2) functions, but then
//   we'd maybe have to do them in groups of __m256's, which basically means
//   a[0..7],b[0..7], etc
//   also have to think about how to arrange things because we output the masks
//   first and then do the selects
//

#include <stdint.h>
#include <immintrin.h>
#include <xmmintrin.h>

#include <iostream>

#include "util.h"

// 32-bit aligned a/b, mask, and results array, do selection for count (which must be a multiple of 8)
void selectf(float *a, float *b, uint32_t *mask, float *res, size_t count)
{
  __m256 *ap = (__m256*)a;
  __m256 *bp = (__m256*)b;
  __m256 *maskp = (__m256*)mask;
  __m256 *resp  = (__m256*)res;

  for(size_t i = 0 ; i < (count >> 3) ; ++i) {
    *resp++ = _mm256_blendv_ps(*ap++, *bp++, *maskp++);
  }
}

// 32-bit aligned a/b, mask, and results array, do selection for count (which must be a multiple of 4)
void selectf2(float *a, float *b, uint32_t *mask, float *res, size_t count)
{
  __m128 *ap = (__m128*)a;
  __m128 *bp = (__m128*)b;
  __m128 *maskp = (__m128*)mask;
  __m128 *resp  = (__m128*)res;

  for(size_t i = 0 ; i < (count >> 2) ; ++i) {
    *resp++ = _mm_blendv_ps(*ap++, *bp++, *maskp++);
  }
}

// this is the traditional decision stump 
void selectslow(float *a, float *b, float *x, float *y, float *res, size_t count)
{
  for(size_t i = 0 ; i < count ; ++i) {
    if(a[i] >= b[i]) {
      res[i] = x[i];
    } else {
      res[i] = y[i];
    }
  }
}

// An attempt to speed up selectslow by grouping the data to improve
// cache access:
//   ie: a0,b0,x0,y0,res0,a1,b1,x1,y1,res1, etc
//
#pragma pack(push,1)
//struct alignas(32) selectentry {
// NOTE: cache use has a huge effect -- doing alignas(32) (which forces the struct to be
// 32 bytes in size, as well), results in about 4x slower performance, probably due to
// memory bandwidth constraints
struct selectentry {
  float a,b,x,y,res;
};
static_assert(sizeof(selectentry) == 20, "Size of selectentry is not 20 bytes");
#pragma pack(pop)
  
void selectlessslow(selectentry *entries, size_t count)
{
  for(size_t i = 0 ; i < count ; ++i) {    
    if(entries[i].a >= entries[i].b) {
      entries[i].res = entries[i].x;
    } else {
      entries[i].res = entries[i].y;
    }
  }
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
void selectf2_2(selectf2_2_entry *entries, __m128 *res, size_t count)
{
  for(size_t i = 0 ; i < (count >> 2) ; ++i) {
    *res++ = _mm_blendv_ps(entries->a, entries->b, entries->mask);
    ++entries;
  }
}

int main(int argc, char **argv)
{
  const size_t TRIALS = 200;
  
  //  alignas(32) float a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  //  alignas(32) float b[16] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
  const size_t COUNT = 8*100000;
  __m256 *a = new __m256[COUNT/8];
  __m256 *b = new __m256[COUNT/8];
  for(size_t i = 0 ; i < COUNT ; ++i) {
    a[i/8][i%8] = i;
    b[i/8][i%8] = COUNT-i;    
  }
  __m256 *mask = new __m256[COUNT/8]; //{ 0xffffffff,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0xffffffff};
  mask[0][0] = 0xffffffff;
  mask[5][0] = 0xffffffff;  
  __m256 *res  = new __m256[COUNT/8];
  //  alignas(32) float res[16];

  if(argv[1][0] == 'a') {
    mask[1][0] = 0xffffffff;
  }

  __m256 *x = new __m256[COUNT/8];
  __m256 *y = new __m256[COUNT/8];

  selectentry *entries = new selectentry[COUNT];
  selectf2_2_entry *entries2 = new selectf2_2_entry[COUNT/4]; // /4 because using __m128 internally
  
  std::cout << "Running tests on " << COUNT << " elements" << std::endl;

  auto timer = []<typename FUNC>(FUNC f_, size_t trials_, const std::string &name_) {
    int64_t start,end;
    double total = 0.0;
    for(size_t trial = 0 ; trial < trials_ ; ++trial) {
      start = get_ts();
      f_();
      end = get_ts();
      total += (end - start);
    }
    total /= trials_;
    std::cout << (uint64_t)total << " nanos/trial (" << trials_ << " trials) for " << name_ << std::endl;
  };

  timer([&](){ selectf(&a[0][0],&b[0][0],(uint32_t*)(&mask[0][0]),&res[0][0],COUNT); }, TRIALS, "selectf");
  timer([&](){ selectslow(&a[0][0],&b[0][0],&x[0][0],&y[0][0],&res[0][0],COUNT); }, TRIALS, "selectslow");
  timer([&](){ selectlessslow(&entries[0],COUNT); }, TRIALS, "selectlessslow");  
  timer([&](){ selectf2(&a[0][0],&b[0][0],(uint32_t*)(&mask[0][0]),&res[0][0],COUNT); }, TRIALS, "selectf2");
  timer([&](){ selectf2_2(&entries2[0],(__m128*)(&res[0][0]),COUNT); }, TRIALS, "selectf2_2");

  float sum = 0;
  for(size_t i = 0 ; i < COUNT; ++i) {
    sum += res[COUNT/8][COUNT%8];
  }
  return sum; // 'a' should == 149, 'b' == 136
}
