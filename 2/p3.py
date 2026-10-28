import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataloader import HW2Dataset
from models import RandomKitchenSinks, MLP
from plotting import plot_surface_2d

ds = HW2Dataset.from_csv("train.csv", "val.csv", "test.csv")
Xtr, ytr, Xv, yv, Xte = ds.train.X, ds.train.y, ds.val.X, ds.val.y, ds.test.X
mse = lambda m, X, y: float(np.mean((m.predict(X) - y) ** 2))
print(Xtr.shape, Xv.shape, Xte.shape, "var(y_val)=", yv.var())

# (a) data visualization
fig = plt.figure(figsize=(11, 4.5))
ax = fig.add_subplot(1, 2, 1)
sc = ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, s=14, cmap="viridis"); plt.colorbar(sc, label="y")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title("Training data (colour = y)")
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.scatter(Xtr[:, 0], Xtr[:, 1], ytr, c=ytr, s=8, cmap="viridis")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y"); ax.set_title("3D view")
plt.tight_layout(); plt.savefig("figs/p3_data.png", dpi=200); plt.close()

def sweep_plot(vals, tr, va, xlabel, name, log=True):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, v, t in zip(axs, (tr, va), ("Training MSE", "Validation MSE")):
        a.plot(vals, v, "o-"); a.set_xlabel(xlabel); a.set_ylabel(t); a.set_title(t)
        if log: a.set_xscale("log")
        a.set_yscale("log")
    plt.tight_layout(); plt.savefig(f"figs/{name}.png", dpi=200); plt.close()

# (b) RKS: sweep sigma with D=2000, lambda=1e-3 (also report sweeps on D and lambda as sanity checks)
res = {}
def rks_sweep(param, vals, **fixed):
    tr, va = [], []
    for v in vals:
        kw = dict(D=2000, sigma=0.5, ridge_lambda=1e-3); kw.update(fixed); kw[param] = v
        m = RandomKitchenSinks(**kw).fit(Xtr, ytr)
        tr.append(mse(m, Xtr, ytr)); va.append(mse(m, Xv, yv))
    return tr, va
sig = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
tr, va = rks_sweep("sigma", sig)
sweep_plot(sig, tr, va, "sigma", "p3_rks_sigma")
best_sig = sig[int(np.argmin(va))]
res["rks_sigma_sweep"] = dict(sigma=sig, train=tr, val=va, best=best_sig)
print("RKS sigma sweep", list(zip(sig, np.round(va, 4))), "best", best_sig)
Ds = [20, 50, 100, 200, 500, 1000, 2000, 4000]
tr, va = rks_sweep("D", Ds, sigma=best_sig)
res["rks_D_sweep"] = dict(D=Ds, train=tr, val=va); print("RKS D sweep", list(zip(Ds, np.round(va, 4))))
sweep_plot(Ds, tr, va, "D (number of features)", "p3_rks_D")
lams = [1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100]
tr, va = rks_sweep("ridge_lambda", lams, sigma=best_sig)
res["rks_lambda_sweep"] = dict(lam=lams, train=tr, val=va); print("RKS lambda sweep", list(zip(lams, np.round(va, 4))))
sweep_plot(lams, tr, va, "ridge lambda", "p3_rks_lambda")
best_D = Ds[int(np.argmin(rks_sweep("D", Ds, sigma=best_sig)[1]))]
best_lam = 1e-3

# (c) MLP: sweep hidden dim (lr=1e-3, bs=128, relu, 300 epochs)
hs = [2, 4, 8, 16, 32, 64, 128, 256, 512]
tr, va = [], []
for h in hs:
    m = MLP(d_in=2, d_hidden=h).fit(Xtr, ytr)
    tr.append(mse(m, Xtr, ytr)); va.append(mse(m, Xv, yv)); print("MLP h", h, tr[-1], va[-1], flush=True)
sweep_plot(hs, tr, va, "hidden dimension", "p3_mlp_hidden")
best_h = hs[int(np.argmin(va))]
res["mlp_hidden_sweep"] = dict(h=hs, train=tr, val=va, best=best_h)

# (d) refit best models; report val MSE; test predictions; surfaces
rks = RandomKitchenSinks(D=best_D, sigma=best_sig, ridge_lambda=best_lam).fit(Xtr, ytr)
mlp = MLP(d_in=2, d_hidden=best_h).fit(Xtr, ytr)
res["final"] = dict(rks=dict(D=best_D, sigma=best_sig, lam=best_lam, val_mse=mse(rks, Xv, yv), train_mse=mse(rks, Xtr, ytr)),
                    mlp=dict(h=best_h, val_mse=mse(mlp, Xv, yv), train_mse=mse(mlp, Xtr, ytr)),
                    val_var=float(yv.var()))
print(res["final"])
np.savetxt("rks_test_predictions.csv", rks.predict(Xte), header="y_pred", comments="")
np.savetxt("mlp_test_predictions.csv", mlp.predict(Xte), header="y_pred", comments="")
plot_surface_2d(rks.predict, (-3, 3), (-3, 3), title="RKS fitted surface", filename="figs/p3_rks_surface.png")
plot_surface_2d(mlp.net, (-3, 3), (-3, 3), title="MLP fitted surface", filename="figs/p3_mlp_surface.png")
json.dump(res, open("results/p3.json", "w"), indent=1)
