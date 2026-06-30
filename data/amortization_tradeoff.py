"""Honest quality-vs-amortization tradeoff (floor-fixed). best-of-16, all 8 robots, %OPT."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["coverage-blind\n(no feature)", "surrogate-gain\n(noisy feature)",
          "true-gain\n(exact feature)", "OPT\n(full search)"]
pct = [80, 80, 96, 100]
# green = fully amortized (no per-robot raycast table at inference); orange = needs table; navy = OPT
colors = ["#27ae60", "#27ae60", "#e67e22", "#2c3e50"]
amort = ["no inference\nraycast table", "no inference\nraycast table",
         "needs per-robot\ntable (~13s)", "full search\n(~68s)"]

fig, ax = plt.subplots(figsize=(9.5, 5.6))
bars = ax.bar(labels, pct, color=colors, width=0.62)
for b, p, a in zip(bars, pct, amort):
    ax.text(b.get_x() + b.get_width() / 2, p + 1.0, f"{p}%", ha="center", fontweight="bold")
    ax.text(b.get_x() + b.get_width() / 2, 6, a, ha="center", fontsize=8, color="white")
ax.axhline(96, color="#e67e22", ls=":", alpha=0.6)
ax.set_ylabel("% of OPT (honest, best-of-16, all 8 robots)")
ax.set_ylim(0, 105)
ax.set_title("Quality vs amortization (floor-fixed): the EXACT marginal-gain feature drives 80% -> 96%,\n"
             "but it is the only variant needing a per-robot raycast table at inference",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("data/amortization_tradeoff.png", dpi=130)
print("saved data/amortization_tradeoff.png")
