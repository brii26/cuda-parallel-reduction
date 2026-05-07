"""
Generate benchmark plots from results.csv produced by ./reduction.
Output: timing_comparison.png, speedup.png, gpu_breakdown.png
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import os

CSV = "results.csv"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(CSV)
except FileNotFoundError:
    print(f"Error: {CSV} not found. Run ./reduction first.")
    sys.exit(1)

n = df["n"].values

# Grafik 1: n vs waktu (log-log)
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(n, df["cpu_mean_ms"], yerr=df["cpu_std_ms"],
            marker="o", label="Serial (CPU)", linewidth=1.8, capsize=4)
ax.errorbar(n, df["gpu_total_mean_ms"], yerr=df["gpu_total_std_ms"],
            marker="s", label="Paralel GPU (total)", linewidth=1.8, capsize=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Ukuran n", fontsize=12)
ax.set_ylabel("Waktu eksekusi (ms)", fontsize=12)
ax.set_title("Perbandingan Waktu: Serial vs Paralel GPU", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "timing_comparison.png"), dpi=150)
print("Saved timing_comparison.png")
plt.close()

# Grafik 2: n vs speedup
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n, df["speedup"], marker="^", color="tab:green",
        linewidth=1.8, markersize=8, label="Speedup")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Speedup = 1x")
ax.set_xscale("log")
ax.set_xlabel("Ukuran n", fontsize=12)
ax.set_ylabel("Speedup (T_serial / T_GPU)", fontsize=12)
ax.set_title("Speedup Paralel GPU vs Serial CPU", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "speedup.png"), dpi=150)
print("Saved speedup.png")
plt.close()

# Grafik 3: stacked bar breakdown GPU
labels = [f"{v:,}" for v in n]
h2d    = df["h2d_mean_ms"].values
kernel = df["kernel_mean_ms"].values
d2h    = df["d2h_mean_ms"].values

x = np.arange(len(labels))
width = 0.5

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x, h2d,    width, label="H2D memcpy",  color="tab:blue")
ax.bar(x, kernel, width, bottom=h2d,           label="Kernel",     color="tab:orange")
ax.bar(x, d2h,    width, bottom=h2d + kernel,  label="D2H memcpy", color="tab:red")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_xlabel("Ukuran n", fontsize=12)
ax.set_ylabel("Waktu (ms)", fontsize=12)
ax.set_title("Breakdown Waktu GPU per Komponen", fontsize=13)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gpu_breakdown.png"), dpi=150)
print("Saved gpu_breakdown.png")
plt.close()
