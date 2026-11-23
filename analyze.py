"""Analyse raw.jsonl: law of demand, elasticities, Engel curves, GARP, consistency.
Usage: python analyze.py results/raw.jsonl [outdir=results]
Writes tables (.tex/.csv), figures (png) and results/macros.tex (numbers used by main.tex).
Zero quantities: the log-log elasticities (own, cross, income) drop zero answers. Section 6 adds zero_share,
corner_share and own_price_mean (elasticity from the log of the mean quantity, zeros included) as checks.
"""
import json, sys, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import GOODS, GOOD_SHORT, PERSONAS, OWN_MULTS, INCOME_MULTS, N_GARP

RAW = sys.argv[1] if len(sys.argv) > 1 else "results/raw.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(f"{OUT}/figs", exist_ok=True); os.makedirs(f"{OUT}/tables", exist_ok=True)
PORDER = list(PERSONAS)
rng = np.random.default_rng(7)

# ------------------------------------------------------------------ load
rows = [json.loads(l) for l in open(RAW)]
df_all = pd.DataFrame(rows)
n_calls, n_fail = len(df_all), int((~df_all.ok).sum())
df = df_all[df_all.ok].copy()
for g in GOODS:
    df["q_" + g] = df.q.map(lambda d: d[g]); df["p_" + g] = df.prices.map(lambda d: d[g])
P = df[["p_" + g for g in GOODS]].values; Q = df[["q_" + g for g in GOODS]].values
df["spend"] = (P * Q).sum(1); df["adh"] = df.spend / df.budget
for g in GOODS: df["w_" + g] = df["p_" + g] * df["q_" + g] / df.spend
MODELS = sorted(df.model.unique())
short = lambda m: m.split(":", 1)[1] if ":" in m else m


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0: return np.nan
    return np.polyfit(x, y, 1)[0]


def boot_ci(sub, xcol, ycol, B=300):
    """Bootstrap 95% CI of the log-log slope, resampling reps (consumers)."""
    reps = sub.rep.unique()
    if len(reps) < 3: return (np.nan, np.nan)
    est = []
    for _ in range(B):
        pick = rng.choice(reps, len(reps))
        s = pd.concat([sub[sub.rep == r] for r in pick])
        est.append(slope(s[xcol], s[ycol]))
    return tuple(np.nanpercentile(est, [2.5, 97.5]))


def latex(dfx, caption, label, fmt="{:.2f}", index=True):
    d = dfx.copy()
    cols = ["l"] + ["r"] * (d.shape[1] - (0 if index else 1)) if index else ["l"] * d.shape[1]
    out = [r"\begin{table}[!htbp]\centering\small", rf"\caption{{{caption}}}\label{{{label}}}",
           r"\begin{tabular}{" + "".join(cols) + "}", r"\hline"]
    hdr = ([""] if index else []) + [str(c).replace("_", " ") for c in d.columns]
    out.append(" & ".join(hdr) + r" \\ \hline")
    for i, r in d.iterrows():
        cells = ([str(i).replace("_", " ")] if index else []) + [
            ((f"{int(v)}" if c in ("consumers", "calls", "failed_calls", "answers_dropped") else fmt.format(v)) if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else ("--" if pd.isna(v) else str(v)))
            for c, v in zip(d.columns, r.values)]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def save(dfx, name, caption, fmt="{:.2f}", index=True):
    dfx.to_csv(f"{OUT}/tables/{name}.csv")
    open(f"{OUT}/tables/{name}.tex", "w").write(latex(dfx, caption, f"tab:{name}", fmt, index))


# ------------------------------------------------------------------ 0. data quality
dq = df.groupby("model").agg(calls=("ok", "size"), mean_budget_use=("adh", "mean"),
                             sd_budget_use=("adh", "std")).rename(index=short)
dq["failed_calls"] = df_all.groupby("model").apply(lambda d: int((~d.ok).sum()), include_groups=False).rename(index=short)
dq["share_within_5pct"] = df.groupby("model").adh.apply(lambda s: ((s - 1).abs() <= 0.05).mean()).rename(index=short)
save(dq, "data_quality", "Data quality: calls made, failures, and how closely the chosen bundle exhausts the budget.")

# ------------------------------------------------------------------ 1. own-price (law of demand)
own = df[df.type == "own"].copy()
base = df[df.type == "base"]
recs_e, recs_ld = [], []
for m in MODELS:
    for per in PORDER:
        for g in GOODS:
            a = own[(own.model == m) & (own.persona == per) & (own.g == g)]
            b = base[(base.model == m) & (base.persona == per)]
            s = pd.concat([a[["rep", "mult", "q_" + g]].rename(columns={"q_" + g: "q"}),
                           b.assign(mult=1.0)[["rep", "mult", "q_" + g]].rename(columns={"q_" + g: "q"})])
            zero_share = (s.q <= 1e-9).mean()
            s = s[s.q > 1e-9].copy()
            s["lp"] = np.log(s.mult); s["lq"] = np.log(s.q)
            e = slope(s.lp, s.lq)
            lo, hi = boot_ci(s, "lp", "lq")
            recs_e.append(dict(model=m, persona=per, good=g, elasticity=e, lo=lo, hi=hi, zero_share=zero_share))
            # per-consumer monotonicity of the demand schedule (5% tolerance)
            ok = []
            for r in sorted(set(s.rep)):
                t = s[s.rep == r].groupby("mult").q.mean().sort_index()
                if len(t) >= 4: ok.append(bool((np.diff(t.values) <= 0.05 * t.values[:-1]).all()))
            recs_ld.append(dict(model=m, persona=per, good=g, monotone=np.mean(ok) if ok else np.nan,
                                giffen=bool(e > 0) if not np.isnan(e) else np.nan))
E = pd.DataFrame(recs_e); LD = pd.DataFrame(recs_ld)
E.to_csv(f"{OUT}/tables/own_price_full.csv", index=False)
t = E.groupby(["model", "good"]).elasticity.mean().unstack()[GOODS].rename(columns=GOOD_SHORT).rename(index=short)
save(t, "own_price", "Own-price elasticity of demand (log-log slope, mean over the four income personas). Negative values obey the law of demand.")
tp = E.pivot_table(index=["model", "persona"], columns="good", values="elasticity")[GOODS].rename(columns=GOOD_SHORT)
tp.index = [f"{short(a)} / {b}" for a, b in tp.index]
save(tp, "own_price_by_persona", "Own-price elasticity by model and persona.")
ld = LD.groupby("model").agg(share_goods_neg=("giffen", lambda s: 1 - s.mean()), consumers_monotone=("monotone", "mean")).rename(index=short)
save(ld, "law_of_demand", "Law of demand: share of (persona, good) cells with a negative elasticity, and share of simulated consumers whose demand schedule never rises with price.")

# ------------------------------------------------------------------ 2. cross-price
recs_c = []
for m in MODELS:
    for gi in GOODS:
        for gj in GOODS:
            if gi == gj: continue
            a = own[(own.model == m) & (own.g == gi)]
            b = base[base.model == m].assign(mult=1.0)
            s = pd.concat([a, b])
            s = s[s["q_" + gj] > 1e-9]
            # demean by persona so that income level does not drive the slope
            s = s.assign(lp=np.log(s.mult), lq=np.log(s["q_" + gj]))
            s["lq"] = s.lq - s.groupby("persona").lq.transform("mean")
            recs_c.append(dict(model=m, price_of=gi, qty_of=gj, cross=slope(s.lp, s.lq)))
C = pd.DataFrame(recs_c)
C.to_csv(f"{OUT}/tables/cross_price_full.csv", index=False)
for m in MODELS:
    t = C[C.model == m].pivot(index="qty_of", columns="price_of", values="cross").reindex(index=GOODS, columns=GOODS)
    t = t.rename(index=GOOD_SHORT, columns=GOOD_SHORT)
    save(t, "cross_price_" + re.sub(r"[^A-Za-z0-9]+", "_", short(m)).strip("_"),
         f"Cross-price elasticities for {short(m).replace('_',' ')}: row = quantity demanded, column = good whose price changes. Positive = substitute, negative = complement.")

open(f"{OUT}/tables/cross_price_all.tex", "w").write("\n".join(open(f"{OUT}/tables/{f}").read() for f in sorted(os.listdir(f"{OUT}/tables")) if f.startswith("cross_price_") and f.endswith(".tex") and f != "cross_price_all.tex"))

# ------------------------------------------------------------------ 3. income / Engel
inc = df[df.type.isin(["income", "base"])].copy()
inc["bm"] = np.where(inc.type == "base", 1.0, inc.bmult)
recs_i, recs_s = [], []
for m in MODELS:
    for per in PORDER:
        s0 = inc[(inc.model == m) & (inc.persona == per)]
        for g in GOODS:
            s = s0[s0["q_" + g] > 1e-9]
            recs_i.append(dict(model=m, persona=per, good=g,
                               income_elasticity=slope(np.log(s.bm), np.log(s["q_" + g]))))
        sh = s0.groupby("bm")[["w_" + g for g in GOODS]].mean()
        for bm, r in sh.iterrows():
            recs_s.append(dict(model=m, persona=per, bm=bm, **{g: r["w_" + g] for g in GOODS}))
I = pd.DataFrame(recs_i); S = pd.DataFrame(recs_s)
I.to_csv(f"{OUT}/tables/income_full.csv", index=False)
t = I.groupby(["model", "good"]).income_elasticity.mean().unstack()[GOODS].rename(columns=GOOD_SHORT).rename(index=short)
save(t, "income", "Income elasticity from Engel curves (log-log slope of quantity on budget, mean over personas). Below 0 = inferior, 0 to 1 = necessity, above 1 = luxury.")
# Engel's law: staples budget share falls with income within persona
engel = S.groupby("model").apply(lambda d: np.mean([np.polyfit(np.log(x.bm), x.staples, 1)[0] < 0 for _, x in d.groupby("persona")]), include_groups=False)
engel_across = df[df.type == "base"].groupby(["model", "persona"]).w_staples.mean().unstack()[PORDER].rename(index=short)
save(engel_across, "staples_share_by_persona", "Budget share of staples at baseline prices, by persona (low to high income). Engel's law predicts a falling share.")

# ------------------------------------------------------------------ 4. GARP / CCEI
def garp(Pm, Xm, e=1.0):
    """Afriat: x_i R0 x_j if e*p_i.x_i >= p_i.x_j. Returns number of GARP violations."""
    n = len(Pm); exp = np.einsum("ij,ij->i", Pm, Xm)
    cost = Pm @ Xm.T                                   # cost[i,j] = p_i . x_j
    R = (e * exp[:, None] >= cost - 1e-9)
    np.fill_diagonal(R, True)
    for k in range(n):                                  # Warshall transitive closure
        R = R | (R[:, [k]] & R[[k], :])
    strict = cost < e * exp[:, None] - 1e-9             # p_i.x_j < e*p_i.x_i : x_i strictly preferred to x_j
    # violation: x_i R x_j and x_j strictly R0 x_i
    viol = R & strict.T
    np.fill_diagonal(viol, False)
    return int(viol.sum())


def ccei(Pm, Xm):
    if garp(Pm, Xm, 1.0) == 0: return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(25):
        mid = (lo + hi) / 2
        if garp(Pm, Xm, mid) == 0: lo = mid
        else: hi = mid
    return lo


g_df = df[df.type == "garp"]
recs_g = []
for (m, per, rep), d in g_df.groupby(["model", "persona", "rep"]):
    if len(d) < N_GARP: continue
    Pm = d[["p_" + g for g in GOODS]].values.astype(float); Xm = d[["q_" + g for g in GOODS]].values.astype(float)
    recs_g.append(dict(model=m, persona=per, rep=rep, violations=garp(Pm, Xm), ccei=ccei(Pm, Xm)))
G = pd.DataFrame(recs_g)
G.to_csv(f"{OUT}/tables/garp_full.csv", index=False)
gt = G.groupby("model").agg(consumers=("ccei", "size"), exact_garp=("violations", lambda s: (s == 0).mean()),
                            mean_ccei=("ccei", "mean"), median_ccei=("ccei", "median"),
                            share_ccei_ge_95=("ccei", lambda s: (s >= 0.95).mean())).rename(index=short)
save(gt, "garp", "Rationality: GARP consistency of simulated consumers over 20 budget sets. CCEI = Afriat critical cost efficiency index (1 = fully rational).")
gp = G.pivot_table(index="model", columns="persona", values="ccei", aggfunc="mean")[PORDER].rename(index=short)
save(gp, "ccei_by_persona", "Mean CCEI by persona.")

# Robustness only (the main GARP result above uses all data): rerun on the same consumers after
# dropping answers that spend outside 0.5-2 times the budget, e.g. empty bundles or large overspends.
SPEND_LO, SPEND_HI = 0.5, 2.0
recs_gr = []
for (m, per, rep), d in g_df.groupby(["model", "persona", "rep"]):
    if len(d) < N_GARP: continue
    kept = d[d.adh.between(SPEND_LO, SPEND_HI)]
    Pm = kept[["p_" + g for g in GOODS]].values.astype(float); Xm = kept[["q_" + g for g in GOODS]].values.astype(float)
    recs_gr.append(dict(model=m, persona=per, rep=rep, answers_kept=len(kept), answers_dropped=len(d) - len(kept),
                        violations=garp(Pm, Xm), ccei=ccei(Pm, Xm)))
GR = pd.DataFrame(recs_gr)
GR.to_csv(f"{OUT}/tables/garp_robust_full.csv", index=False)
gr = GR.groupby("model").agg(consumers=("ccei", "size"), answers_dropped=("answers_dropped", "sum"),
                             exact_garp=("violations", lambda s: (s == 0).mean()), mean_ccei=("ccei", "mean"),
                             median_ccei=("ccei", "median"), share_ccei_ge_95=("ccei", lambda s: (s >= 0.95).mean())).rename(index=short)
save(gr, "garp_robust", "Robustness: GARP and CCEI after dropping budget-set answers that spend less than 0.5 or more than 2 times the budget. The main result (all answers) is in the GARP table.")

# ------------------------------------------------------------------ 5. consistency across repeated runs
cv = base.groupby(["model", "persona"])[["q_" + g for g in GOODS]].agg(lambda s: s.std() / s.mean() if s.mean() > 0 else np.nan)
cv.columns = [GOOD_SHORT[c[2:]] for c in cv.columns]
cvt = cv.groupby("model").mean().rename(index=short)
save(cvt, "variability", "Coefficient of variation of the chosen quantity across repeated identical prompts (mean over personas).")

# ------------------------------------------------------------------ 6. zeros and corner solutions
ZERO = 1e-9
by_mp = df.groupby(["model", "persona"])
zs = by_mp[["q_" + g for g in GOODS]].agg(lambda s: (s <= ZERO).mean())
zs.columns = [GOOD_SHORT[c[2:]] for c in zs.columns]
zs = zs.reindex([(m, p) for m in MODELS for p in PORDER]).dropna(how="all")
zs.index = [f"{short(a)} / {b}" for a, b in zs.index]
save(zs, "zero_share", "Share of answers (all scenario types) that buy zero units of each good, by model and persona.")

df["max_share"] = df[["w_" + g for g in GOODS]].max(axis=1)
df["corner"] = df.max_share > 0.8
cs = df.pivot_table(index="model", columns="persona", values="corner", aggfunc="mean").reindex(columns=PORDER)
cs.insert(0, "all", df.groupby("model").corner.mean())
cs = cs.rename(index=short)
save(cs, "corner_share", "Corner solutions: share of answers where a single good takes more than 80\\% of actual spending, by model and persona.")


def mean_q_slope(table):
    """Log-log slope of the mean quantity at each price multiplier (zeros included in the mean).
    table: reps x multipliers. Multipliers whose mean is zero or missing are dropped."""
    means = table.mean(axis=0)
    means = means[means > ZERO]
    return slope(np.log(means.index.values.astype(float)), np.log(means.values)), len(means)


def ci95(draws):
    draws = np.asarray(draws, float)
    return tuple(np.nanpercentile(draws, [2.5, 97.5])) if len(draws) and not np.all(np.isnan(draws)) else (np.nan, np.nan)


rng_mean = np.random.default_rng(8)       # separate stream, so the existing bootstrap draws are unchanged
recs_m, recs_mm = [], []
for m in MODELS:
    reps_m = np.sort(df[df.model == m].rep.unique())
    for g in GOODS:
        tables = {}
        for per in PORDER:
            a = own[(own.model == m) & (own.persona == per) & (own.g == g)]
            b = base[(base.model == m) & (base.persona == per)].assign(mult=1.0)
            s = pd.concat([a[["rep", "mult", "q_" + g]], b[["rep", "mult", "q_" + g]]])
            tables[per] = s.pivot_table(index="rep", columns="mult", values="q_" + g, aggfunc="mean").reindex(reps_m)
        draws = {per: [] for per in PORDER}
        if len(reps_m) >= 3:
            for _ in range(300):
                pick = rng_mean.choice(reps_m, len(reps_m))
                for per in PORDER:
                    draws[per].append(mean_q_slope(tables[per].loc[pick])[0])
        for per in PORDER:
            e, n_pts = mean_q_slope(tables[per])
            lo, hi = ci95(draws[per])
            recs_m.append(dict(model=m, persona=per, good=g, elasticity_mean_q=e, lo=lo, hi=hi, price_points=n_pts))
        avg_draws = np.nanmean(np.array([draws[per] for per in PORDER]), axis=0) if len(reps_m) >= 3 else []
        lo, hi = ci95(avg_draws)
        recs_mm.append(dict(model=m, good=g, ci_lo=lo, ci_hi=hi))
EM = pd.DataFrame(recs_m).merge(E[["model", "persona", "good", "elasticity", "zero_share"]], on=["model", "persona", "good"])
EM = EM.rename(columns={"elasticity": "elasticity_nonzero"})
EM.to_csv(f"{OUT}/tables/own_price_mean_full.csv", index=False)
em = EM.groupby(["model", "good"]).agg(nonzero_only=("elasticity_nonzero", "mean"), mean_quantity=("elasticity_mean_q", "mean"),
                                       zero_share=("zero_share", "mean"))
em = em.join(pd.DataFrame(recs_mm).set_index(["model", "good"]))[["nonzero_only", "mean_quantity", "ci_lo", "ci_hi", "zero_share"]]
em = em.reindex([(m, g) for m in MODELS for g in GOODS])
em.index = [f"{short(a)} / {GOOD_SHORT[b]}" for a, b in em.index]
save(em, "own_price_mean", "Own-price elasticity two ways (mean over personas): from non-zero answers only (as in the main table) "
     "and from the log of the mean quantity at each price, with zeros included. 95\\% CI of the mean-quantity value from a bootstrap over reps. "
     "Zero share = share of answers at these prices that buy none of the good.")

# ------------------------------------------------------------------ figures
colors = plt.cm.tab10.colors
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# fig1: demand curves
fig, axs = plt.subplots(1, len(GOODS), figsize=(13, 2.9), sharey=False)
for ax, g in zip(axs, GOODS):
    for ci, m in enumerate(MODELS):
        a = own[(own.model == m) & (own.g == g)]
        b = base[base.model == m].assign(mult=1.0)
        s = pd.concat([a, b])
        s = s.assign(rel=s["q_" + g] / s.groupby("persona")["q_" + g].transform(lambda x: x[s.loc[x.index, "mult"] == 1.0].mean()))
        mm = s.groupby("mult").rel.mean().sort_index()
        ax.plot(mm.index, mm.values, marker="o", ms=3, color=colors[ci], label=short(m))
    xs = np.array(sorted([1.0] + OWN_MULTS)); ax.plot(xs, 1 / xs, ls=":", color="grey", label="unit elastic")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(GOOD_SHORT[g]); ax.set_xlabel("price / base price")
axs[0].set_ylabel("quantity / base quantity"); axs[-1].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/figs/demand_curves.png", dpi=200); plt.close()

# fig2: Engel curves
fig, axs = plt.subplots(1, len(GOODS), figsize=(13, 2.9))
for ax, g in zip(axs, GOODS):
    for ci, m in enumerate(MODELS):
        s = inc[inc.model == m]
        s = s.assign(rel=s["q_" + g] / s.groupby("persona")["q_" + g].transform(lambda x: x[s.loc[x.index, "bm"] == 1.0].mean()))
        mm = s.groupby("bm").rel.mean().sort_index()
        ax.plot(mm.index, mm.values, marker="o", ms=3, color=colors[ci], label=short(m))
    xs = np.array(sorted([1.0] + INCOME_MULTS)); ax.plot(xs, xs, ls=":", color="grey", label="unit income elastic")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(GOOD_SHORT[g]); ax.set_xlabel("budget / base budget")
axs[0].set_ylabel("quantity / base quantity"); axs[-1].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/figs/engel_curves.png", dpi=200); plt.close()

# fig3: CCEI by model
if len(G):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    data = [G[G.model == m].ccei.values for m in MODELS]
    ax.boxplot(data, tick_labels=[short(m) for m in MODELS], showfliers=True)
    ax.axhline(0.95, ls=":", color="grey"); ax.set_ylabel("CCEI"); ax.set_ylim(0, 1.02)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(f"{OUT}/figs/ccei.png", dpi=200); plt.close()

# fig4: elasticity heatmap
fig, ax = plt.subplots(figsize=(6.2, 0.6 + 0.5 * len(MODELS)))
im = ax.imshow(t_ := E.groupby(["model", "good"]).elasticity.mean().unstack()[GOODS].values, cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
ax.set_xticks(range(len(GOODS)), [GOOD_SHORT[g] for g in GOODS]); ax.set_yticks(range(len(MODELS)), [short(m) for m in MODELS])
for i in range(t_.shape[0]):
    for j in range(t_.shape[1]): ax.text(j, i, f"{t_[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax, label="own-price elasticity"); plt.tight_layout()
plt.savefig(f"{OUT}/figs/elasticity_heatmap.png", dpi=200); plt.close()

# ------------------------------------------------------------------ macros for the paper
M = {
    "NCalls": f"{n_calls:,}", "NFailed": f"{n_fail:,}", "NModels": str(len(MODELS)),
    "NConsumers": f"{G.shape[0] // max(len(MODELS), 1) if len(G) else 0}",
    "LDShare": f"{100 * (1 - LD.giffen.astype(float).mean()):.0f}",
    "LDMonotone": f"{100 * LD.monotone.mean():.0f}",
    "MeanCCEI": f"{G.ccei.mean():.3f}" if len(G) else "--",
    "ExactGARP": f"{100 * (G.violations == 0).mean():.0f}" if len(G) else "--",
    "MeanBudgetUse": f"{100 * df.adh.mean():.1f}",
    "StaplesElast": f"{E[E.good == 'staples'].elasticity.mean():.2f}",
    "EatingOutElast": f"{E[E.good == 'eating_out'].elasticity.mean():.2f}",
    "EntertainmentElast": f"{E[E.good == 'entertainment'].elasticity.mean():.2f}",
    "ElastSpread": f"{E.groupby('good').elasticity.mean().max() - E.groupby('good').elasticity.mean().min():.2f}",
    "EngelHold": f"{100 * engel.mean():.0f}",
    "ZeroAnswerShare": f"{100 * (df[['q_' + g for g in GOODS]] <= ZERO).any(axis=1).mean():.0f}",
    "CornerShare": f"{100 * df.corner.mean():.0f}",
    "MeanQElastSpread": f"{EM.groupby('good').elasticity_mean_q.mean().max() - EM.groupby('good').elasticity_mean_q.mean().min():.2f}",
}
open(f"{OUT}/macros.tex", "w").write("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
print(json.dumps(M, indent=1)); print(t.round(2)); print(gt.round(3)); print(ld.round(3)); print(dq.round(3))
print(em.round(2)); print(cs.round(3))
