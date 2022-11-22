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
// cdef
//  <=
// dcfe
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

// pack entire tree into structure
struct tree2 {
  uint32_t a_splitVarID, c_splitVarID, e_splitVarID;
  float b_splitValue, d_splitValue, f_splitValue;
  float one, two, three, four;
};

std::vector<tree2> forest2;

inline double tree_eval_simd(const tree2 &t1_, const tree2 &t2_, const std::vector<float> &x_)
{
  __m256 cmp1 = _mm256_set_ps(x_[t1_.a_splitVarID],
			      x_[t1_.a_splitVarID],
			      t1_.b_splitValue,
			      t1_.b_splitValue,
			      x_[t2_.a_splitVarID],
			      x_[t2_.a_splitVarID],
			      t2_.b_splitValue,
			      t2_.b_splitValue);
  // note we could probably achieve this with a shuffle
  __m256 cmp2 = _mm256_set_ps(t1_.b_splitValue,
			      t1_.b_splitValue,
			      x_[t1_.a_splitVarID],
			      x_[t1_.a_splitVarID],
			      t2_.b_splitValue,
			      t2_.b_splitValue,
			      x_[t2_.a_splitVarID],
			      x_[t2_.a_splitVarID]);
  __m256 cmpres1 = _mm256_cmp_ps(cmp1, cmp2, 18); // <= 

  cmp1 = _mm256_set_ps(x_[t1_.c_splitVarID],
		       t1_.d_splitValue,		      
		       x_[t1_.e_splitVarID],
		       t1_.f_splitValue,
		       x_[t2_.c_splitVarID],
		       t2_.d_splitValue,		       
		       x_[t2_.e_splitVarID],
		       t2_.f_splitValue);
  cmp2 = _mm256_set_ps(t1_.d_splitValue,
		       x_[t1_.c_splitVarID],		       
		       t1_.f_splitValue,
		       x_[t1_.e_splitVarID],
		       t2_.d_splitValue,
		       x_[t2_.c_splitVarID],		       
		       t2_.f_splitValue,
		       x_[t2_.e_splitVarID]);

  __m256 cmpres2 = _mm256_cmp_ps(cmp1, cmp2, 18); // <=

  __m256i mask = _mm256_and_si256((__m256i)cmpres1, (__m256i)cmpres2);
  __m256 res1 = _mm256_set_ps(t1_.one, t1_.two, t1_.three, t1_.four,
			     t2_.one, t2_.two, t2_.three, t2_.four);
  __m256 res2 = _mm256_setzero_ps();
  _mm256_maskstore_ps((float *)(&res2), mask, res1);
  
  return horizontal_add(res2); // note: the SUM of the two trees!
}

double rf_eval_simd(const std::vector<tree2> &f_, const std::vector<float> &x_)
{
  double total = 0.0;
  size_t count = f_.size() / 2;
  for(size_t i = 0 ; i < count ; i++) {
    total += tree_eval_simd(f_[i*2 + 0], f_[i*2 + 1], x_);
  }
  return total / f_.size();;
}

void compare_rfs(const std::vector<tree *> &f_, const std::vector<tree2> &f2_, const std::vector<float> &x_)
{
  double eps = 0.0000001;
  size_t count = f_.size() / 2;
  for(size_t i = 0 ; i < count ; i++) {
    double v1_f1 = tree_eval(*f_[i*2 + 0], x_);
    double v2_f1 = tree_eval(*f_[i*2 + 1], x_);
    double vtot_f2 = tree_eval_simd(f2_[i*2 + 0], f2_[i*2 + 1], x_);

    if(fabs((v1_f1 + v2_f1) - vtot_f2) > eps) {
      std::abort();
    }
  }
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

  // now, restructure the tree so we can evaluate it with SIMD
  // build a forest with trees of depth 2
  for(size_t t = 0 ; t < NUM_TREES ; ++t) {
    const auto &treep = *(forest[t]);

    tree2 tt = { treep[0].splitVarID, treep[1].splitVarID, treep[2].splitVarID,
		treep[0].splitValue, treep[1].splitValue, treep[2].splitValue,
		treep[3].splitValue, treep[4].splitValue, treep[5].splitValue, treep[6].splitValue };

    forest2.push_back(tt);
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

  // sanity check
  compare_rfs(forest, forest2, x);
  
  timer([&](){ return rf_eval(forest, x); }, TRIALS, "rf_eval");

  timer([&](){ return rf_eval_simd(forest2, x); }, TRIALS, "rf_eval_simd");
  
  return 0;
}
