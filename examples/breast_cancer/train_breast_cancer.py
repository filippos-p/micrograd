import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
 
from micrograd import MLP
 
# --- load data ---
data = load_breast_cancer()
X = data.data          # shape (569, 30)
y = data.target        # 0 = malignant, 1 = benign
 
# --- train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# --- standardize: fit (mu, sigma) on train only, apply to both ---
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
 
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma
 
print("train shape:", X_train.shape, "test shape:", X_test.shape)
print("train mean (should be ~0):", X_train.mean(axis=0)[:3])
print("train std  (should be ~1):", X_train.std(axis=0)[:3])


y_train_mapped = y_train * 2 - 1   
y_test_mapped  = y_test * 2 - 1


n = MLP(30, [16, 1])

EPOCHS = 20      # each epoch takes a while (pure-Python scalar autograd) -- be patient
LR = 0.05

for k in range(EPOCHS):
  ypred = [n(x) for x in X_train]
    
  # mean squared error, averaged over samples so LR doesn't need to
  # change with the size of the dataset
  loss = sum((yout - ygt) ** 2 for ygt, yout in zip(y_train_mapped, ypred)) / len(X_train)
  # backward pass
  n.zero_grad()
  loss.backward()

  # update
  for p in n.parameters():
    p.data += -LR * p.grad

  train_acc = np.mean([
        (1 if yp.data > 0 else -1) == yt
        for yp, yt in zip(ypred, y_train_mapped)
    ])
  print(f"epoch {k:3d}  loss={loss.data:.4f}  train_acc={train_acc:.3f}")

# --- final evaluation on the held-out test set ---
ytest_pred = [n(x) for x in X_test]
test_acc = np.mean([
    (1 if yp.data > 0 else -1) == yt
    for yp, yt in zip(ytest_pred, y_test_mapped)
])
print(f"\nfinal test accuracy: {test_acc:.4f}")
 
