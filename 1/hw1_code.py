"""Homework 1, problems 4 and 5. Run: python hw1_code.py"""
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_default_dtype(torch.float64)

ale = pd.read_csv("data/ale.csv")
X_ale = ale[["anchor_ratio", "trans_range", "node_density", "iterations"]].values
y_ale = ale["ale"].values

# ---------------------------------------------------------------- Problem 4
print("=" * 70, "\nProblem 4: normality of residuals\n" + "=" * 70)
models = {
    "Linear": make_pipeline(StandardScaler(), LinearRegression()),
    "SVR (rbf)": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.01)),
    "Decision tree": DecisionTreeRegressor(max_depth=5, random_state=SEED),
    "Random forest": RandomForestRegressor(n_estimators=200, random_state=SEED),
    "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=5000, random_state=SEED)),
}
fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
for ax, (name, m) in zip(axes, models.items()):
    # out-of-fold predictions so residuals are not distorted by overfitting
    resid = y_ale - cross_val_predict(m, X_ale, y_ale, cv=10)
    k2, p = stats.normaltest(resid)
    print(f"{name:14s} resid mean={resid.mean():+.4f} std={resid.std():.4f} "
          f"skew={stats.skew(resid):+.2f} kurt={stats.kurtosis(resid):+.2f} "
          f"normaltest stat={k2:.2f} p={p:.2e}")
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title(f"{name}\nnormaltest p={p:.1e}")
plt.tight_layout(); plt.savefig("figs/qqplots.png", dpi=150); plt.close()

# ---------------------------------------------------------------- helpers
def sgd_fit(loss_fn, params, Xtr, ytr, lr, epochs, batch=32, seed=SEED):
    """Minibatch SGD with torch.optim.SGD; loss_fn(params, X, y) -> scalar."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(params, lr=lr)
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(params, Xtr[idx], ytr[idx]).backward()
            opt.step()
    return params

def to_t(*arrs):
    return [torch.tensor(a, dtype=torch.float64) for a in arrs]

# ---------------------------------------------------------------- Problem 5: classification
print("\n" + "=" * 70, "\nProblem 5: classification (Breast Cancer Wisconsin)\n" + "=" * 70)
Xc, yc = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=SEED, stratify=yc)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

base = LogisticRegression(max_iter=5000).fit(Xtr_s, ytr)
print(f"Baseline sklearn LogisticRegression test acc: {accuracy_score(yte, base.predict(Xte_s)):.4f}")

Xtr_t, Xte_t, ytr_t = to_t(Xtr_s, Xte_s, ytr.astype(float))
EPS = 1e-7
std_normal = torch.distributions.Normal(0.0, 1.0)

def make_bce(link):
    def loss(params, X, y):
        w, b = params
        f = link(X @ w + b).clamp(EPS, 1 - EPS)          # clip for stability
        return -(y * torch.log(f) + (1 - y) * torch.log(1 - f)).mean()
    return loss

for name, link in [("(a) Logistic (sigmoid)", torch.sigmoid), ("(b) Probit (Normal CDF)", std_normal.cdf)]:
    w = torch.zeros(Xtr_t.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    sgd_fit(make_bce(link), [w, b], Xtr_t, ytr_t, lr=0.1, epochs=200)
    with torch.no_grad():
        pred = (link(Xte_t @ w + b) > 0.5).numpy().astype(int)
    print(f"{name:25s} test acc: {accuracy_score(yte, pred):.4f}")

# ---------------------------------------------------------------- Problem 5: regression
print("\n" + "=" * 70, "\nProblem 5: regression (ALE)\n" + "=" * 70)
Xtr, Xte, ytr, yte = train_test_split(X_ale, y_ale, test_size=0.2, random_state=SEED)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

base = LinearRegression().fit(Xtr_s, ytr)
print(f"Baseline sklearn LinearRegression test MAE: {mean_absolute_error(yte, base.predict(Xte_s)):.4f}")

Xtr_t, Xte_t, ytr_t = to_t(Xtr_s, Xte_s, ytr)
NU = 5.0

def mse_loss(params, X, y):
    w, b = params
    return ((y - (X @ w + b)) ** 2).mean()

def t_nll(params, X, y):
    w, b = params
    r = y - (X @ w + b)
    # -log p(r; nu) = -lgamma((nu+1)/2) + lgamma(nu/2) + 0.5 log(nu pi) + (nu+1)/2 log(1 + r^2/nu)
    return (-torch.lgamma(torch.tensor((NU + 1) / 2)) + torch.lgamma(torch.tensor(NU / 2))
            + 0.5 * np.log(NU * np.pi) + (NU + 1) / 2 * torch.log1p(r ** 2 / NU)).mean()

for name, loss in [("(a) Squared error + SGD", mse_loss), (f"(b) Student-t (nu={NU:g}) NLL + SGD", t_nll)]:
    w = torch.zeros(Xtr_t.shape[1], requires_grad=True)
    b = torch.tensor([float(ytr.mean())], requires_grad=True)   # start at mean of y
    sgd_fit(loss, [w, b], Xtr_t, ytr_t, lr=0.05, epochs=300)
    with torch.no_grad():
        mae = (Xte_t @ w + b - torch.tensor(yte)).abs().mean().item()
    print(f"{name:36s} test MAE: {mae:.4f}   w={w.detach().numpy().round(3)} b={b.item():.3f}")
