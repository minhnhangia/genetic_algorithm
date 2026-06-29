"""48h exploration figure: budget frontier (quality vs inference search budget) + per-robot LNS vs OPT."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- budget frontier: MULTI-SEED (3 seeds x 7 robots), mean %OPT +/- CI ---
# HONEST (one-sided opaque floor + z>=0 pool); supersedes the floor-bug numbers.
Ns = [1, 8, 16, 32, 64]
excl = [85, 95, 97, 98, 99]          # mean % of OPT
excl_ci = [7.1, 2.4, 2.1, 1.5, 1.4]  # 95% CI in % points

# --- per-robot policy best-of-32 vs OPT (HONEST, floor-fixed) ---
robots = ["kairos", "robout", "summit_xl", "summit_steel", "theron", "theron+top", "watcher", "vogui"]
lns = [0.3130, 0.2835, 0.3056, 0.3151, 0.3051, 0.3010, 0.3029, 0.2934]  # policy best-of-32
opt = [0.3142, 0.3406, 0.3260, 0.3148, 0.3019, 0.2980, 0.3133, 0.2999]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

# Panel 1: multi-seed budget frontier with CIs
ax1.errorbar(Ns, excl, yerr=excl_ci, fmt="o-", color="#27ae60", lw=2, capsize=4,
             label="7 robots (excl rbrobout), 3 seeds")
ax1.axhline(84, color="#e67e22", ls=":", label="prior SOTA (true-gain bo16, 84%)")
for n, v in zip(Ns, excl):
    ax1.annotate(f"{v}%", (n, v), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#1e8449")
ax1.set_xscale("log", base=2); ax1.set_xticks(Ns); ax1.set_xticklabels(Ns)
ax1.set_xlabel("inference search budget  N  (best-of-N true-verified evals)")
ax1.set_ylabel("% of OPT (multi-start+LS ceiling)")
ax1.set_title("Cost / quality frontier (multi-seed) — search budget is the lever")
ax1.legend(fontsize=9, loc="lower right"); ax1.grid(alpha=0.3); ax1.set_ylim(65, 100)

# Panel 2: per-robot LNS(49) vs OPT
x = np.arange(len(robots)); w = 0.38
ax2.bar(x - w / 2, opt, w, label="OPT (ceiling)", color="#2c3e50")
ax2.bar(x + w / 2, lns, w, label="policy best-of-32", color="#27ae60")
ax2.set_xticks(x); ax2.set_xticklabels(robots, rotation=30, ha="right")
ax2.set_ylabel("true-verified fitness")
ax2.set_title("Per-robot (honest): all 8 reach 83-101% of OPT (mean 96%)")
ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Honest eval (floor-fixed): amortized policy + modest search budget -> ~98% of OPT, zero-shot",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("data/exploration_results.png", dpi=130)
print("saved data/exploration_results.png")
