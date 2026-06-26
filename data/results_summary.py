"""One-off: render the results-so-far summary figure (true-verified fitness)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

robots = ["kairos", "robout", "summit_xl", "summit_steel",
          "theron", "theron+top", "watcher", "vogui"]

opt      = [0.3098, 0.3404, 0.3273, 0.3317, 0.3034, 0.2947, 0.3254, 0.2999]
baseline = [0.2783, 0.1828, 0.2573, 0.2666, 0.2426, 0.2376, 0.2702, 0.2443]  # coverage-blind, 3-seed
truegain = [0.3131, 0.0953, 0.3069, 0.3078, 0.2777, 0.2840, 0.2879, 0.2609]  # true-raycast gain, 3-seed

x = np.arange(len(robots))
w = 0.26

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2.1, 1]})

# --- Panel 1: per-robot ---
ax1.bar(x - w, opt,      w, label="OPT (multi-start+LS, ceiling)", color="#2c3e50")
ax1.bar(x,     baseline, w, label="Baseline best-of-N (coverage-blind)", color="#3498db")
ax1.bar(x + w, truegain, w, label="True-raycast-gain best-of-N (coverage-aware)", color="#27ae60")
ax1.set_xticks(x); ax1.set_xticklabels(robots, rotation=30, ha="right")
ax1.set_ylabel("true-verified fitness")
ax1.set_title("Per-robot zero-shot fitness vs the OPT ceiling")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(axis="y", alpha=0.3)
ax1.axhline(np.mean(opt), color="#2c3e50", ls=":", alpha=0.5)

# --- Panel 2: fleet progression with CI ---
methods = ["greedy\n(argmax)", "SIL\n(rejected)", "Surrogate-gain\n(rejected)", "Baseline\nbest-of-N", "True-gain\nbest-of-N", "OPT\nceiling"]
means   = [0.1805, 0.2437, 0.2378, 0.2474, 0.2667, 0.3166]
errs    = [0.0330, 0.0185, 0.0140, 0.0139, 0.0287, 0.0]
colors  = ["#95a5a6", "#c0392b", "#e67e22", "#3498db", "#27ae60", "#2c3e50"]
xb = np.arange(len(methods))
ax2.bar(xb, means, 0.6, yerr=errs, capsize=5, color=colors)
for i, m in enumerate(means):
    ax2.text(i, m + 0.006, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks(xb); ax2.set_xticklabels(methods, fontsize=8.5)
ax2.set_ylabel("fleet mean fitness (±95% CI)")
ax2.set_title("Fleet progression")
ax2.set_ylim(0, 0.36)
ax2.grid(axis="y", alpha=0.3)
# annotate % of OPT
for i, m in enumerate(means):
    ax2.text(i, 0.01, f"{100*m/0.3166:.0f}%", ha="center", fontsize=8, color="white", fontweight="bold")

fig.suptitle("Cross-robot zero-shot LiDAR placement — results so far (true-verified)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "data/results_summary.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
print(f"fleet: greedy=0.180  baseline={np.mean(baseline):.3f}  "
      f"true-gain={np.mean(truegain):.3f}  OPT={np.mean(opt):.3f}")
