//
// a test program for 2-level decision trees
//
// the idea is based on what vectest.cc does.  in this case it is a little more complex because things are 2-level
//
// similar to gpu programming we can imagine branches being "shut off" if they are not selected
//
// imagine we do a _mm256_cmp_ps, this gives us 8 floating point lanes
// there are 4 possible outcomes from a 2-level decision tree
// lets say the tree looks like this:
/*
         a <= b
        /      \
      c <= d  e <= f
       /  \    /  \
      1    2  3    4
*/
//
// if we stack the comparisons vertically, we could do the following:
//
// aabb
//  <=
// bbaa
//  && 
// cedf
//  <=
// dfce
//
// So, we have two sets of comparisons that generate masks, and at the end
// we take the bitwise && of the two masks -- only one possibility will be 1.
//
// Since we have 8 lanes and only use 4, this means we can evaluate two trees at once.
//
// It's possible there is a terminal node, in which case we could replicate the parent node
// (since if we visit the terminal node, the parent condition must have evaluated to true)
//
// This implies we should store the trees differently than the way they are usually stored.
// It would be best to store a,b,c,d and 1,2,3,4 all in the same structure.
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

struct node
{
  uint64_t leftChildNodeID;
  uint64_t rightChildNodeID;
  uint64_t splitVarID;
  float splitValue;              
};

typedef std::vector<node> tree;
std::vector<tree *> forest;

double tree_eval(const tree &t_, const std::vector<float> &x_)
{
  // For each sample start in root, drop down the tree and return final value
  size_t nodeID = 0;
  while (1) {
    const auto &tn = t_[nodeID];
    // Break if terminal node
    if (tn.leftChildNodeID == 0 && tn.rightChildNodeID == 0) {
      break;
    }

    // Move to child
    double value = x_[tn.splitVarID];

    if (value <= tn.splitValue) {
      // Move to left child
      nodeID = tn.leftChildNodeID;
    } else {
      // Move to right child
      nodeID = tn.rightChildNodeID;
    }
  }

  return t_[nodeID].splitValue;
}

double rf_eval(const std::vector<tree *> &f_, const std::vector<float> &x_)
{
  double total = 0.0;
  for(const auto &t : f_) {
    total += tree_eval(*t, x_);
  }
  return total / f_.size();
}

int main(int argc, char **argv)
{
  const size_t TRIALS = 200;
  const size_t NUM_PREDS = 256; // we'll consider 256 possible predictors
  const size_t NUM_TREES = 500000; // with 500000 trees
  
  std::mt19937_64 g(1234);      // mersenne twister with constant seed for reproducibility
  std::uniform_real_distribution<float> d(-0.1, 0.1);
  
  // build a forest with trees of depth 2
  for(size_t t = 0 ; t < NUM_TREES ; ++t) {
    auto *treep = new tree;
    treep->push_back({1, 2, g() % NUM_PREDS, d(g) }); // node 0
    treep->push_back({3, 4, g() % NUM_PREDS, d(g) }); // node 1
    treep->push_back({5, 6, g() % NUM_PREDS, d(g) }); // node 2
    treep->push_back({0, 0, 0, d(g) }); // terminal node 3
    treep->push_back({0, 0, 0, d(g) }); // terminal node 4
    treep->push_back({0, 0, 0, d(g) }); // terminal node 5
    treep->push_back({0, 0, 0, d(g) }); // terminal node 6    
    forest.push_back(treep);
  }

  // generate predictors
  std::vector<float> x(NUM_PREDS);
  for(size_t i = 0 ; i < NUM_PREDS ; ++i) {
    x[i] = d(g);
  }
    
  std::cout << "Running " << TRIALS << " trials on forest with " << NUM_TREES << " trees of depth=2" << std::endl;

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

  timer([&](){ return rf_eval(forest, x); }, TRIALS, "rf_eval");

  return 0;
}
