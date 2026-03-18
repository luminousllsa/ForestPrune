import json
import numpy as np
import matplotlib.pyplot as plt

json_path = "./collect_prune_stats.json"

with open(json_path, "r") as f:
    stats = json.load(f)

# [T, R]
keep_region_ratio = np.array(stats["keep_region_ratio"], dtype=float)

T, R = keep_region_ratio.shape
mean_keep_ratio = keep_region_ratio.mean(axis=0)   # [R]

print("=== 每个区域的平均保留率 ===")
for r, v in enumerate(mean_keep_ratio):
    print(f"Region {r}: {v:.4f}")

# 1) 柱状图：整体每个区域平均保留率
plt.figure(figsize=(8, 4))
x = np.arange(R)
plt.bar(x, mean_keep_ratio)
plt.xticks(x, [f"R{r}" for r in range(R)])
plt.ylabel("Keep Ratio")
plt.xlabel("Region")
plt.title("Average Keep Ratio per Region")
for i, v in enumerate(mean_keep_ratio):
    plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("./region_keep_ratio_bar.png", dpi=300)
plt.show()

# 2) 热力图：每帧-每区域 保留率
plt.figure(figsize=(10, 6))
im = plt.imshow(keep_region_ratio, aspect="auto")
plt.colorbar(im, label="Keep Ratio")
plt.xlabel("Region")
plt.ylabel("Frame")
plt.title("Per-frame Keep Ratio per Region")
plt.xticks(np.arange(R), [f"R{r}" for r in range(R)])
plt.yticks(np.arange(T))
plt.tight_layout()
plt.savefig("./region_keep_ratio_heatmap.png", dpi=300)
plt.show()