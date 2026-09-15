"""
ILLUSTRATIVE / SYNTHETIC ablation generator  --  DIAGNOSTIC USE ONLY.
These numbers are drawn from a generative model. They are NOT measurements and
MUST NOT be placed in the manuscript. Purpose: show what internally-consistent
ablation data looks like, so real checkpoint numbers can be checked against it.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(20260709)
SEEDS = 5
CONDS = ["full", "w/o IMW", "w/ log-spec", "w/o SSL"]

# --- generative model -------------------------------------------------------
# value[arch,cond,seed] = base + true_effect[cond] + shared_seed_effect[seed]*s + iid_noise
# The SHARED seed effect is what makes the paired design powerful: it inflates
# the marginal per-condition SD but cancels in the paired difference.
# Two architectures get INDEPENDENT draws -> their observed offsets differ.

# base 'full' means and the seed/noise scales, chosen near the paper's values
META = {
 "FM":  {"Spec":(1.50,0.06,0.015), "SC":(0.0299,0.0016,0.0005), "High":(0.89,0.06,0.02)},
 "DX7": {"Spec":(57.0,4.0,0.25),   "SC":(0.850,0.030,0.004),    "High":(7.95,0.28,0.08)},
 "TAL": {"Spec":(0.168,0.010,0.003),"SC":(0.400,0.018,0.005),   "High":(214.,13.,3.5)},
}
# TRUE effects added to 'full' (direction matches the paper: SSL biggest, IMW best).
# NOTE these are modest and honest: the IMW-vs-logspec Spec gap is deliberately
# small so the illustration shows a *borderline* effect, not an all-significant fantasy.
EFF = {  # multiplied by the metric's seed-scale to keep units sane
 "Spec": {"full":0.0, "w/o IMW":0.45, "w/ log-spec":0.28, "w/o SSL":1.15},
 "SC":   {"full":0.0, "w/o IMW":0.60, "w/ log-spec":0.34, "w/o SSL":1.30},
 "High": {"full":0.0, "w/o IMW":1.9,  "w/ log-spec":1.15, "w/o SSL":3.6},
}

def gen(arch_shift):
    out = {}
    for dset, mets in META.items():
        for met,(base,seed_sd,iid) in mets.items():
            shared = rng.normal(0, seed_sd, SEEDS)          # per-seed difficulty (shared across conds)
            col = {}
            for c in CONDS:
                eff = EFF[met][c]*seed_sd*0.28 + arch_shift*seed_sd*0.15
                col[c] = base + eff + shared + rng.normal(0, iid, SEEDS)
            out[(dset,met)] = col
    return out

trans = gen(arch_shift=+0.4)   # independent architecture draws => different offsets
unet  = gen(arch_shift=-0.4)

def fmt(m,met):
    if met=="SC":  return f"{m:.4f}"
    if met=="Spec" and m<1: return f"{m:.3f}"
    if met=="Spec": return f"{m:.2f}"
    return f"{m:.2f}"

print("="*78)
print("ILLUSTRATIVE / SYNTHETIC -- NOT FOR THE MANUSCRIPT")
print("="*78)
for name,blk in [("Transformer",trans),("U-Net",unet)]:
    print(f"\n### {name}  (mean +/- SD over 5 synthetic seeds)")
    print(f"{'Cond':12s}"+"".join(f"{d+' '+m:>14s}" for d in META for m in META[d]))
    for c in CONDS:
        row=f"{c:12s}"
        for d in META:
            for m in META[d]:
                v=blk[(d,m)][c]; row+=f"{fmt(v.mean(),m)+'±'+fmt(v.std(ddof=1),m):>14s}"
        print(row)

print("\n"+"="*78)
print("CROSS-ARCHITECTURE OFFSET CHECK (this is what should DIFFER, and here does)")
print("="*78)
for d in META:
    for m in META[d]:
        tf=trans[(d,m)]["full"].mean(); uf=unet[(d,m)]["full"].mean()
        line=f"{d} {m:5s}: "
        for c in ["w/o IMW","w/ log-spec","w/o SSL"]:
            dt=trans[(d,m)][c].mean()-tf; du=unet[(d,m)][c].mean()-uf
            line+=f"{c}(T{dt:+.3f}/U{du:+.3f}) "
        print(line)

print("\n"+"="*78)
print("PAIRED SIGNIFICANCE (Transformer): full vs each condition, N=5, Holm per metric")
print("shows: marginal SD can be large yet PAIRED diff is what determines significance")
print("="*78)
for d in META:
    for m in META[d]:
        base=trans[(d,m)]["full"]
        print(f"\n{d} {m}  (full marginal SD={base.std(ddof=1):.4g})")
        res=[]
        for c in ["w/o IMW","w/ log-spec","w/o SSL"]:
            x=trans[(d,m)][c]; diff=x-base
            t,p=stats.ttest_rel(x,base)
            dz=diff.mean()/diff.std(ddof=1)
            res.append((c,diff.mean(),diff.std(ddof=1),t,p,dz))
        # Holm within this metric family
        order=sorted(range(len(res)),key=lambda i:res[i][4])
        adj={}; k=len(res)
        for rank,i in enumerate(order):
            adj[i]=min(1.0,(k-rank)*res[i][4])
        for i,(c,md,sd,t,p,dz) in enumerate(res):
            star="*" if adj[i]<0.05 else " "
            print(f"  {c:12s} Δ={md:+.4g}  pairedSD={sd:.4g}  t(4)={t:+.2f}  p={p:.4f}  Holm={adj[i]:.4f}{star}  dz={dz:+.2f}")
