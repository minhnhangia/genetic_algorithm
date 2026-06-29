"""Honest cost figure from measured data (floor-fixed, ff ckpts). No recompute."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

names = ["kairos", "robout", "summit_xl", "summit_steel", "theron", "theron+top", "watcher", "vogui"]
t_pol = np.array([1.75, 1.31, 2.08, 2.46, 1.68, 1.71, 1.74, 1.72])
f_pol = np.array([0.3130, 0.2835, 0.3056, 0.3151, 0.3051, 0.3010, 0.3029, 0.2934])
t_opt = np.array([42.13, 71.62, 97.71, 63.68, 66.64, 67.53, 71.70, 60.77])
f_opt = np.array([0.3142, 0.3406, 0.3260, 0.3148, 0.3019, 0.2980, 0.3133, 0.2999])
x = np.arange(len(names)); w = 0.38

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
ax1.bar(x - w / 2, t_opt, w, label="OPT (multi-start + LS)", color="#2c3e50")
ax1.bar(x + w / 2, t_pol, w, label="Policy (best-of-32)", color="#27ae60")
ax1.set_yscale("log"); ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha="right")
ax1.set_ylabel("wall-clock to produce a layout (s, log)")
ax1.set_title(f"Inference cost per robot — policy ~{np.mean(t_opt / t_pol):.0f}x cheaper than OPT")
ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3, which="both")

cp, qp = t_pol.mean(), 100 * (f_pol / f_opt).mean()
co = t_opt.mean()
ax2.scatter([co], [100], s=180, color="#2c3e50", label="OPT (ceiling)", zorder=3)
ax2.scatter([cp], [qp], s=180, color="#27ae60", label="Policy best-of-32", zorder=3)
ax2.annotate(f"OPT\n{co:.0f}s, 100%", (co, 100), textcoords="offset points", xytext=(-12, -28),
             fontsize=9, ha="center")
ax2.annotate(f"Policy\n{cp:.1f}s, {qp:.0f}% OPT", (cp, qp), textcoords="offset points",
             xytext=(12, 6), fontsize=9, color="#1e8449")
ax2.set_xlabel("wall-clock per robot (s)"); ax2.set_ylabel("% of OPT quality (all 8 robots)")
ax2.set_title("Cost vs quality — near-optimal at a fraction of the cost")
ax2.grid(alpha=0.3); ax2.legend(fontsize=9, loc="lower right"); ax2.set_ylim(min(qp - 6, 90), 102)

fig.suptitle("Amortized DRL policy vs OPT (honest, floor-fixed): inference cost",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("data/cost_comparison.png", dpi=130)
print(f"saved data/cost_comparison.png  (policy {cp:.1f}s vs OPT {co:.0f}s = {co/cp:.0f}x, {qp:.0f}% OPT)")
