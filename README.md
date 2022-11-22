# Speeding Up Decision Stumps with SIMD

Random forests are often the most degenerate case in terms of computation performance when it comes to prediction time.
We have to traverse the trees, comparing various predictors against split values, until we come to a terminal node and output
its value.  It involves branching, has poor cache performance, and is practically unoptimizable.

The simple approach to speeding up random forests is to evaluate groups of them in parallel.  Indeed, for large enough numbers of
trees this can scale reasonably well.  

Generally, random forests (of a large enough size) are not practical for realtime applications due to the cost of evaluation, even
when parallelized.  However, we might be able to make a trade-off in model quality for a fairly large speedup in prediction time.
This is where decision stumps come in.

[Decision stumps](https://en.wikipedia.org/wiki/Decision_stump) are a variant of random forest, with maximum tree depth set to 1.
Whether or not this is a useful machine learning algorithm for classification or prediction for a particular problem is not for us to say.
It appears to be the case in the literature that acceptable results can be obtained for certain data sets.

The general decision stump forest evaluation code looks like:

```
tot = 0;
for(size_t i = 0 ; i < count ; ++i) {
  if(a[i] <= b[i]) {
    tot += x[i];
  } else {
    tot += y[i];
  }
}
return tot / count;
```

As it turns out, we can do this using SIMD instructions and:

- Avoid branching entirely
- Do the computation in parallel

We make use of two SIMD instrinsics:

- `_mm_cmp_ps` : this compares multiple floating-point values at once
- `_mm_blendv_ps` : this acts as a selector, given a mask

We also use SIMD add instructions to accumulate our results (with some final bit-twiddling to perform a "horizontal add" on our accumulated total).

If we build `vectest` (via `make`), we can get a basic benchmark as follows:

```
nolanw@Adrenalin ~/speedstumps $ ./exec/opt/vectest
Running tests on 800000 elements
4000578 nanos/trial (200 trials) for selectslow (val=-5.52614e-05)
644030 nanos/trial (200 trials) for selectf (val=-5.5262e-05)
707964 nanos/trial (200 trials) for selectf2 (val=-5.52618e-05)
```

`selectslow` is the baseline evaluation function.  We get about a 6x speedup for the 256-bit `selectf` variant.  For completeness' sake we also implement the 128-bit version as `selectf2`.

Given that the theoretical max bandwith increase we could achieve is 8x (256 bits / 32 bits per float) a value over 6x seems reasonable.  Since each core has SIMD units, theoretically evaluating groups of stumps in parallel should further boost evaluation throughput.

# But Can We Do It For Trees w/Depth=2?

As it turns out, yes!

Similar to gpu programming we can imagine branches being "shut off" if they are not selected.
Imagine we do a `_mm256_cmp_ps`, this gives us 8 floating point lanes.
There are 4 possible outcomes from a 2-level decision tree.
Let's say the tree looks like this:

```
         a <= b
        /      \
      c <= d  e <= f
       /  \    /  \
      1    2  3    4
```

If we stack the comparisons vertically, we could do the following:

```
   aabb
    <=
   bbaa
    && 
   cdef
    <=
   dcfe
```

So, we have two sets of comparisons that generate masks, and at the end
we take the bitwise && of the two masks -- only one possibility will be 1.

Since we have 8 lanes and only use 4, this means we can evaluate two trees at once.

This implies we should store the trees differently than the way they are usually stored.
It would be best to store a,b,c,d and 1,2,3,4 all in the same structure.

If we build the vectest2 program, which compares a traditional random forest storage/evaluation loop
with our SIMD version, we get a little over 2x speedup:

```
nolanw@Adrenalin ~/speedstumps (depth2) $ ./exec/opt/vectest2
Running 200 trials on forest with 500000 trees of depth=2
13987547 nanos/trial (200 trials) for rf_eval (val=7.81268e-05)
5467814 nanos/trial (200 trials) for rf_eval_simd (val=7.81268e-05)
```

Which is pretty much what we'd hope for.
