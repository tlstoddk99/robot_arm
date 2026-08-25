import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

FIG_DIR = os.path.expanduser("~/robot_arm/isaac_so_arm101/my_scripts/results/figures")

joints = ["shoulder_lift", "elbow_flex", "wrist_flex"]
base_rmse = [2.99, 2.60, 11.69]
tuned_rmse = [1.16, 1.01, 0.91]

x = np.arange(len(joints))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, base_rmse, w, label="base", color="tab:blue", alpha=0.7)
b2 = ax.bar(x + w/2, tuned_rmse, w, label="tuned", color="tab:green")

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)

ax.set_ylabel("Trajectory RMSE (deg)")
ax.set_title("Dynamics Matching: base vs tuned (trajectory RMSE)")
ax.set_xticks(x)
ax.set_xticklabels(joints)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
out = f"{FIG_DIR}/summary_rmse.png"
fig.savefig(out, dpi=90)
print(f"저장: {out}")

reductions = [(1 - t/b) * 100 for b, t in zip(base_rmse, tuned_rmse)]
print("\n=== 종합 ===")
print(f"{'joint':16s} {'base':>8} {'tuned':>8} {'감소율':>8}")
for j, b, t, r in zip(joints, base_rmse, tuned_rmse, reductions):
    print(f"{j:16s} {b:>8.2f} {t:>8.2f} {r:>7.1f}%")
