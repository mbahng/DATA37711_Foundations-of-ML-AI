import json, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
torch.manual_seed(0); np.random.seed(0)
df = pd.read_parquet("taxi_trips.parquet")
feats = [c for c in df.columns if c != "trip_duration"]
X, y = df[feats].values.astype(np.float32), df["trip_duration"].values.astype(np.float32)
idx = np.random.default_rng(0).permutation(len(df)); n = len(df)
tr, va, te = idx[:int(.7*n)], idx[int(.7*n):int(.85*n)], idx[int(.85*n):]
Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]

def nll(mu, s, y): return float(np.mean(0.5*np.log(2*np.pi*s**2) + (y-mu)**2/(2*s**2)))
res = {}
# baselines: (B1) global Gaussian; (B2) linear regression mean + constant sigma
mu0, s0 = ytr.mean(), ytr.std()
res["B1_const"] = dict(nll=nll(mu0, s0, yte), rmse=float(np.sqrt(np.mean((yte-mu0)**2))))
lr = LinearRegression().fit(Xtr, ytr); s1 = np.std(ytr - lr.predict(Xtr))
res["B2_linear"] = dict(nll=nll(lr.predict(Xte), s1, yte), rmse=float(np.sqrt(np.mean((yte-lr.predict(Xte))**2))))

class Net(nn.Module):
    def __init__(s, d=7, h=128, linear=False):
        super().__init__()
        s.g = nn.Linear(d, 2) if linear else nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,2))
    def forward(s, x):
        o = s.g(x); return o[:,0], nn.functional.softplus(o[:,1]) + 1e-3   # link: identity for mu, softplus for sigma
def loss_fn(mu, s, y): return (torch.log(s) + (y-mu)**2/(2*s**2)).mean()   # Gaussian NLL (up to const)
T = lambda a: torch.from_numpy(a)

def fit(linear):
    torch.manual_seed(0)
    net = Net(linear=linear); opt = torch.optim.Adam(net.parameters(), 1e-3)
    best, bad, state = 1e9, 0, None
    for ep in range(100):
        net.train(); p = torch.randperm(len(Xtr))
        for i in range(0, len(p), 1024):
            b = p[i:i+1024]; opt.zero_grad(); loss_fn(*net(T(Xtr[b])), T(ytr[b])).backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = loss_fn(*net(T(Xva)), T(yva)).item()
        if v < best - 1e-4: best, bad, state = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
        print(ep, round(v,4), flush=True)
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): mu, s = [t.numpy() for t in net(T(Xte))]
    return mu, s, ep+1

mu3, s3, ep3 = fit(linear=True)    # B3: linear mean + softplus(linear) sigma
res["B3_linear_softplus"] = dict(nll=nll(mu3, s3, yte), rmse=float(np.sqrt(np.mean((yte-mu3)**2))), epochs=ep3)
mu, s, ep = fit(linear=False)
res["MLP"] = dict(nll=nll(mu, s, yte), rmse=float(np.sqrt(np.mean((yte-mu)**2))), epochs=ep)

# (d,e): P(Y<45) = Phi((45-mu)/sigma)
lab = (yte < 45).astype(float)
def evalp(p, name):
    p = np.clip(p, 1e-12, 1-1e-12)
    return dict(brier=float(np.mean((p-lab)**2)), logloss=float(-np.mean(lab*np.log(p)+(1-lab)*np.log(1-p))), acc=float(np.mean((p>.5)==lab)))
res["prob"] = dict(base_rate=float(lab.mean()),
    MLP=evalp(norm.cdf((45-mu)/s), "mlp"), B1_const=evalp(np.full(len(yte), norm.cdf((45-mu0)/s0)), "b1"),
    B2_linear=evalp(norm.cdf((45-lr.predict(Xte))/s1), "b2"),
    B3_linear_softplus=evalp(norm.cdf((45-mu3)/s3), "b3"),
    empirical_const=evalp(np.full(len(yte), (ytr<45).mean()), "emp"))
res["sigma_stats"] = dict(min=float(s.min()), median=float(np.median(s)), max=float(s.max()), const_B2=float(s1))
res["sigma_stats"]["corr_mu_sigma"] = float(np.corrcoef(mu, s)[0,1])
res["sigma_stats"]["corr_y_sigma"] = float(np.corrcoef(yte, s)[0,1])
res["sizes"] = [len(tr), len(va), len(te)]
print(json.dumps(res, indent=1)); json.dump(res, open("results/p1.json","w"), indent=1)
