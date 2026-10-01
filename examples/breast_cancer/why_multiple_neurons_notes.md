# Why a layer needs multiple neurons

In our example with the MLP architecture, we used a `[3,4,4,1]` schema:
`inputs (3) -> hidden layer 1 (4) -> hidden layer 2 (4) -> output (1)`.

I was wondering why we need more than one neuron, let alone multiple
layers. So — what happens for *one* neuron when we "pass" the inputs
through it?

A neuron takes `x1, x2, x3`, computes `sum(wi*xi) + b`, and outputs
one number. So it's a function:

```
f: R^n -> R^m
```

For one neuron with 3 inputs: domain `R^n` = `R^3` (where our inputs
`x1,x2,x3` live), codomain `R^m` = `R^1` (one neuron = one output
number).

```
f(x) = sum(wi*xi) = w1x1 + w2x2 + w3x3 = w·x
```

For the function `f` we have the **rank**, which is the dimension of
its image. The rank of `f` is bounded:

```
rank(f) ≤ min(n, m)
```

where `n` = dim(domain), `m` = dim(codomain). For one neuron:
`rank(f) ≤ min(3, 1) = 1`. The inequality is strict (rank falls below
`min(n,m)`) only when there are hidden linear dependencies between
weight vectors — with a single neuron that doesn't apply, so
`rank(f) = 1` exactly (as long as the weights aren't all zero).

For a layer we stack each neuron's weights as a row of a matrix `W`.
In general `W` is a matrix with `m` rows (number of neurons = the
codomain dimension) and `n` columns (number of inputs = the domain
dimension):

```
f(x) = Wx,   x = (x1, x2, x3)
```

So `rank(f) = rank(W)` — not an analogy, the layer's linear part *is*
this matrix multiplication.

We use Gaussian elimination to find the number of non-zero,
independent rows of `W`. Each independent row lets us solve for
exactly one variable in terms of the others, and that variable
becomes a **pivot variable**. The number of pivot variables is the
rank of `f`. Any non-pivot variable becomes a **free variable**.

```
ker(f) = { x ∈ R^n | f(x) = 0 }
```

From the rank-nullity theorem:

```
dim(ker f) = dim(domain) − rank(f) = n − rank(f)
```

If `dim(ker f) > 0`, there exist genuinely different inputs `x ≠ x'`
(e.g. two tumors with genuinely different measurements) such that
`f(x) = f(x')` — they collapse to the exact same number after being
passed into one neuron.

To actually draw this, drop down to 2 inputs instead of 3 (`f: R² ->
R¹`), since a plane is easier to sketch than R³. With one equation
`w1x1 + w2x2 = 0` and two unknowns, the kernel is a whole **line**
through the origin, not just the origin itself — pick any `x2`
freely, solve for `x1`, you're on the kernel. Two different points on
that same line (different coordinate pairs, both satisfying the
equation) both map to `f(x) = 0` — different samples, same output.

More generally, every line *parallel* to the kernel line is a level
set `w1x1 + w2x2 = c` for some constant `c` — every point on that
line maps to the same output `c`, for the same reason: it's one
equation, one condition, with a whole line's worth of solutions.

![Points on the same line collapse to one output; the kernel is the line through the origin where f=0](docs/kernel_collapse_r2_to_r1.svg)

Each **linearly independent** equation (= each neuron whose weight vector
isn't just a multiple of another neuron's) adds one independent constraint, 
reducing the dimension of the solution space by one. Equivalently, 
after row reduction, each independent constraint produces one pivot.
With 2 independent equations in R² (2 neurons,
non-parallel weight vectors), the only point satisfying *both at the
same time* is the intersection of two non-parallel lines through the
origin — which is just the origin itself:

![Two independent equations intersect only at the origin; distinct inputs stay distinct](docs/full_rank_r2_to_r2.svg)

If `dim(ker f) = 0`, the system has the trivial solution `{0}` only —
so there aren't infinite solutions for every line in the kernel,
because the kernel *is* `{0}`. Now different samples `x` map to
different outputs: **f becomes injective**.

From rank-nullity, for `dim(ker f) = 0` we need `rank(f) = dim(domain)
= n`. So we need the output space (codomain) to be at least as big as
the input space — i.e. **at least `n` neurons** — in order not to lose
any information structurally, before training even starts.
