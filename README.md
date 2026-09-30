# micrograd

A tiny scalar-valued autograd engine, implemented from scratch, plus a
minimal multi-layer perceptron (MLP) built on top of it. No PyTorch,
no NumPy — just Python and `math`. Based on
[Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

## What this is

Every number in a computation is wrapped in a `Value`, which remembers
two things: the operation that produced it, and which other `Value`s
(its "children") went into that operation. This turns a sequence of
arithmetic into a **computation graph**.

Calling `.backward()` on the final output of that graph computes the
gradient of the output with respect to every `Value` it depends on —
this is backpropagation, and it's exactly how gradients are computed
in real neural networks (just at a much smaller scale here).

## Why not just write one big symbolic formula and differentiate it?

Because for any graph beyond a handful of operations, the symbolic
formula explodes — and every value that gets reused in multiple places
would force you to re-derive shared sub-expressions by hand.

Instead, `.backward()` never builds a symbolic expression. It computes
a **numeric local derivative at each node** (e.g. `d(a*b)/da = b`,
`d(a+b)/da = 1`) and combines it with the gradient already flowing
in from downstream, via the chain rule:

```
grad(node) = upstream_grad × local_derivative
```

This keeps every step small and constant-cost, regardless of how deep
the graph is.

## Why gradients accumulate, and why `.backward()` needs a topological sort

If a `Value` is used in more than one place (e.g. `a` appears in both
`c = a*b` and `d = c + a`), it receives a separate gradient contribution
from *each* path that uses it — and those contributions are **added**,
not overwritten. That's why every backward pass does `self.grad += ...`
rather than `self.grad = ...`.

This also means a node's gradient is only *complete* once every child
that depends on it has already run its own backward step. `.backward()`
handles this by first building a **topological ordering** of the graph
(every node placed after all of its children), then walking that order
in reverse. This guarantees that by the time a node's gradient is read,
every path that feeds into it has already contributed.

## Why `zero_grad()` matters

`.grad` is accumulated with `+=`, not reset automatically. If you run
a new forward/backward pass without zeroing gradients first, the new
gradient gets added on top of the stale one from the *previous* point
in parameter space — producing a value that doesn't correspond to the
gradient at either point. `zero_grad()` (called on `Neuron`/`Layer`/`MLP`)
resets every parameter's `.grad` to `0.0` before each new `.backward()`
call.

## Structure

```
micrograd/
├── engine.py      Value: the autograd engine (+, *, **, tanh, exp, backward)
├── nn.py          Neuron, Layer, MLP (built on Value)
└── visualize.py   optional: draw the computation graph with graphviz
example.py         trains the toy MLP on the 4-sample dataset
```

## Usage

```python
from micrograd import MLP

n = MLP(3, [4, 4, 1])   # 3 inputs -> 4 -> 4 -> 1 output
ypred = n([2.0, 3.0, -1.0])
```

Training loop (gradient descent, minimizing loss ⇒ `-=`):

```python
for k in range(steps):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    n.zero_grad()
    loss.backward()

    for p in n.parameters():
        p.data += -0.05 * p.grad
```

See `example.py` for a runnable version.
