# Binary classification on the Breast Cancer Wisconsin dataset

A from-scratch MLP (using the `micrograd` engine — no PyTorch, no
sklearn models) trained to classify tumors as malignant or benign,
from 30 real-valued features (`sklearn.datasets.load_breast_cancer`).

This was the first time the toy 4-sample network from the micrograd
video was pointed at a real dataset (569 samples, 30 features), and
it broke in two separate, instructive ways.

## Setup

- **Standardization**: each of the 30 features is on a wildly
  different scale (e.g. `mean area` ranges into the thousands,
  `mean smoothness` is ~0.05–0.16). Fed raw into a neuron's weighted
  sum, the large-scale features would dominate both the output and
  the gradients, regardless of which features actually matter. Every
  feature is rescaled to mean 0 / std 1: `(x - mu) / sigma`, with
  `mu` and `sigma` computed **only on the training set** and then
  applied to both train and test — computing them on the full dataset
  would leak test-set information into training (lookahead bias).
- **Labels remapped from `{0,1}` to `{-1,+1}`**: the network's output
  layer uses `tanh`, whose range is `(-1, 1)`. Leaving labels as
  `{0,1}` is asymmetric: label `0` is trivially reachable (`tanh(0)=0`),
  while label `1` can only be approached, never reached, pushing the
  network to keep growing weights indefinitely for that class alone.
  Remapping to `{-1,+1}` makes both classes equally reachable only in
  the limit, which is symmetric and much better behaved.
- **80/20 train/test split**, evaluated on held-out test data the
  network never trained on.

## Two bugs found by actually running it on real data

### 1. Saturated from the very first step (bad weight initialization)

The toy example uses 3 inputs; this dataset has 30. `Neuron.__init__`
originally drew every weight from `Uniform(-1, 1)` regardless of how
many inputs the neuron has. With 30 inputs, the pre-activation
(`act = w·x + b`) is a sum of 30 roughly-independent terms — and the
standard deviation of a sum of independent terms grows with
`sqrt(n)`, not `n`. Checked directly: with `nin=3` this stays small,
but with `nin=30`, `act` was already ~14 at random initialization,
before any training. `tanh(14) ≈ 1` with a derivative of essentially
`0` — the network was fully saturated at step zero, so gradients were
~0 and no weight ever moved.

**Fix**: scale each weight by `1/sqrt(nin)` at initialization, so the
pre-activation's scale stays reasonable regardless of how many inputs
a neuron has (and bias initialized to `0` instead of random, to not
add to the problem).

### 2. Weights exploding mid-training (loss summed instead of averaged)

The toy example sums the loss over 4 samples; this dataset has 455
training samples. Every weight is reused across all 455 per-sample
forward passes, and gradients accumulate (`+=`) across every one of
those paths — correctly, since that's exactly what the gradient of a
summed loss over a batch is. But that means the total gradient scales
up with the number of samples: summing over 455 samples instead of 4
produces gradients roughly two orders of magnitude larger. With a
learning rate (`0.05`) tuned for the 4-sample case, this produced a
single wildly oversized update, pushing some weight's pre-activation
far enough that `tanh`'s internal `exp(2x)` overflowed.

Important distinction worked out during debugging: the accumulated
`p.grad` itself is not "wrong" — it's the correct gradient of the
summed loss. The problem is that this gradient's *scale* depends on
the number of samples, which a fixed learning rate doesn't account
for. And raising float precision wouldn't fix it either — even without
any overflow, a large enough update can overshoot the minimum and
increase the loss instead of decreasing it. The overflow was just
where this particular implementation happened to crash first, not the
root problem itself.

**Fix**: divide the summed loss by the number of samples (mean
instead of sum), so its scale — and the resulting gradients — stay
independent of dataset size, and a fixed learning rate behaves
consistently regardless of how many samples are in the batch.

## Result

After 20 full-batch gradient descent steps (`EPOCHS=20`, `LR=0.05`):
loss dropped from ~1.29 to ~0.17, **97.4% accuracy on the held-out
test set**.

## Notes

- Training is full-batch (every epoch runs a forward pass over the
  entire training set before one `backward()` + update) — reasonable
  at this dataset size, but noted as a place where mini-batch/SGD
  would later replace it.
- This is a pure-Python, scalar-by-scalar autograd engine — no
  vectorization — so each epoch over 455×30 inputs takes several
  seconds. Expected, not a bug.
