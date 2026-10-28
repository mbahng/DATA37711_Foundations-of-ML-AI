import json, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)
df = pd.read_csv("retail.csv", dtype={"StockCode": str, "CustomerID": str})
res = {"cardinality": df[["StockCode", "CustomerID"]].nunique().to_dict(), "n": len(df)}
print(res)
df["ly"] = np.log1p(df.Quantity)           # target transform for heavy tail
idx = np.random.default_rng(0).permutation(len(df)); n = len(df)
tr, va, te = df.iloc[idx[:int(.7*n)]], df.iloc[idx[int(.7*n):int(.85*n)]], df.iloc[idx[int(.85*n):]]
cats = ["StockCode", "CustomerID"]
# integer codes; 0 = unseen
codes = {}
for c in cats:
    m = {k: i+1 for i, k in enumerate(tr[c].unique())}; codes[c] = m
enc = lambda d: np.stack([d[c].map(codes[c]).fillna(0).astype(int).values for c in cats], 1)
Ctr, Cva, Cte = enc(tr), enc(va), enc(te)
res["unseen_test_frac"] = {c: float((Cte[:, i] == 0).mean()) for i, c in enumerate(cats)}
te_enc = TargetEncoder(cols=cats, smoothing=10).fit(tr[cats], tr.ly)
Ttr, Tva, Tte = [te_enc.transform(d[cats]).values.astype(np.float32) for d in (tr, va, te)]
ytr, yva, yte = [d.ly.values.astype(np.float32) for d in (tr, va, te)]
nc = [len(codes[c]) + 1 for c in cats]
# out-of-fold (cross-fitted) target encoding for the training rows (val/test use the encoder fit on all of train)
Ttr_oof = np.zeros_like(Ttr)
for a, b in KFold(5, shuffle=True, random_state=0).split(tr):
    e = TargetEncoder(cols=cats, smoothing=10).fit(tr.iloc[a][cats], tr.iloc[a].ly)
    Ttr_oof[b] = e.transform(tr.iloc[b][cats]).values

class Net(nn.Module):
    def __init__(s, use_emb, use_te, ed=16):
        super().__init__(); s.ue, s.ut = use_emb, use_te
        s.emb = nn.ModuleList([nn.Embedding(k, ed) for k in nc]) if use_emb else None
        d = (2*ed if use_emb else 0) + (2 if use_te else 0)
        s.f = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, c, t):
        z = []
        if s.ue: z += [e(c[:, i]) for i, e in enumerate(s.emb)]
        if s.ut: z.append(t)
        return s.f(torch.cat(z, 1)).squeeze(1)

def run(use_emb, use_te, wd=1e-4, oof=False):
    torch.manual_seed(SEED)
    net = Net(use_emb, use_te); opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=wd)
    T = lambda C, Tt, y: (torch.from_numpy(C), torch.from_numpy(Tt), torch.from_numpy(y))
    D = [T(Ctr, Ttr_oof if oof else Ttr, ytr), T(Cva, Tva, yva), T(Cte, Tte, yte)]
    best, bad, st = 1e9, 0, None
    for ep in range(60):
        net.train(); p = torch.randperm(len(ytr))
        for i in range(0, len(p), 512):
            b = p[i:i+512]; opt.zero_grad()
            nn.functional.huber_loss(net(D[0][0][b], D[0][1][b]), D[0][2][b], delta=1.0).backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = nn.functional.mse_loss(net(*D[1][:2]), D[1][2]).item()
        if v < best - 1e-4: best, bad, st = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6: break
    net.load_state_dict(st); net.eval()
    with torch.no_grad(): pred = net(*D[2][:2]).numpy()
    return pred, net

def metrics(pred, mask=None):
    m = np.ones(len(yte), bool) if mask is None else mask
    q, qp = np.expm1(yte[m]), np.expm1(np.clip(pred[m], 0, None))
    return dict(n=int(m.sum()), rmse_log=float(np.sqrt(np.mean((pred[m]-yte[m])**2))), mae_log=float(np.mean(np.abs(pred[m]-yte[m]))),
                mae_qty=float(np.mean(np.abs(q-qp))), medae_qty=float(np.median(np.abs(q-qp))))
both_seen = (Cte > 0).all(1)
res["baseline_mean"] = metrics(np.full(len(yte), ytr.mean()))
res["baseline_mean_some_unseen"] = metrics(np.full(len(yte), ytr.mean()), ~both_seen)
res["baseline_te_only_lookup"] = metrics(Tte.mean(1))
preds = {}
for name, (e, t, o) in {"target_enc": (False, True, False), "embedding": (True, False, False), "combined": (True, True, False), "target_enc_oof": (False, True, True), "combined_oof": (True, True, True)}.items():
    p, net = run(e, t, oof=o); preds[name] = p
    res[name] = dict(all=metrics(p), both_seen=metrics(p, both_seen), some_unseen=metrics(p, ~both_seen))
    print(name, res[name]["all"], flush=True)
    if name == "embedding" and SEED == 0:   # inspect: do embeddings of frequent customers/items vary with the target?
        E = net.emb[0].weight.detach().numpy()[1:]
        cnt = tr.StockCode.map(tr.StockCode.value_counts()).groupby(tr.StockCode).first().reindex(list(codes["StockCode"])).values
        mean_ly = tr.groupby("StockCode").ly.mean().reindex(list(codes["StockCode"])).values
        U = np.linalg.svd(E - E.mean(0), full_matrices=False)
        res["emb_stockcode_pc1_corr_with_mean_logqty"] = float(np.corrcoef(U[0][:, 0]*U[1][0], mean_ly)[0, 1])
res["both_seen_frac"] = float(both_seen.mean())
print(json.dumps(res, indent=1)); json.dump(res, open(f"results/p2_seed{SEED}.json", "w"), indent=1)
