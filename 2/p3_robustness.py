import json, numpy as np
from dataloader import HW2Dataset
from models import RandomKitchenSinks, MLP
ds = HW2Dataset.from_csv("train.csv", "val.csv", "test.csv")
Xtr, ytr, Xv, yv = ds.train.X, ds.train.y, ds.val.X, ds.val.y
mse = lambda m, X, y: float(np.mean((m.predict(X) - y) ** 2))
out = {}
out["rks_sigma0.2_by_seed"] = [mse(RandomKitchenSinks(D=2000, sigma=0.2, ridge_lambda=1e-3, seed=s).fit(Xtr, ytr), Xv, yv) for s in range(5)]
out["rks_sigma0.3_by_seed"] = [mse(RandomKitchenSinks(D=2000, sigma=0.3, ridge_lambda=1e-3, seed=s).fit(Xtr, ytr), Xv, yv) for s in range(5)]
out["rks_final_by_seed"] = [mse(RandomKitchenSinks(D=1000, sigma=6, ridge_lambda=1e-3, seed=s).fit(Xtr, ytr), Xv, yv) for s in range(5)]
out["mlp_h4_by_seed"] = [mse(MLP(d_in=2, d_hidden=4, seed=s).fit(Xtr, ytr), Xv, yv) for s in range(3)]
out["mlp_final_by_seed"] = [mse(MLP(d_in=2, d_hidden=512, seed=s).fit(Xtr, ytr), Xv, yv) for s in range(3)]
for k, v in out.items(): print(k, np.round(v, 4))
json.dump(out, open("results/p3_robustness.json", "w"), indent=1)
