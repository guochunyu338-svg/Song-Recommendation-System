"""實驗：產生報告中所有表格與圖片，結果存到 results/。

實驗 1：選擇分群數 K（Elbow Method＋Silhouette Score）
實驗 2：不同 K 值下，K-means 搜尋與全量搜尋的「速度 vs 準確度」比較
實驗 3：特徵消融（Ablation），觀察各特徵對推薦品質的影響
實驗 4：範例歌曲的推薦結果
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from recommender import (FEATURES_4, FEATURES_8, KMeansRecommender, find_song,
                         load_songs, recommend_exact, scale_features)

TOP_N = 5            # 每次推薦幾首
N_QUERIES = 300      # 隨機抽幾首歌當測試歌曲
SEED = 0
os.makedirs("results", exist_ok=True)

songs = load_songs("songs.csv")
X = scale_features(songs, FEATURES_4)
rng = np.random.default_rng(SEED)
queries = rng.choice(len(songs), N_QUERIES, replace=False)
print(f"清理後歌曲數：{len(songs)}，測試歌曲數：{N_QUERIES}")


def genre_precision(target, rec_idx):
    """推薦歌曲中，與目標歌曲有共同曲風的比例。"""
    g = songs.at[target, "genres"]
    return np.mean([len(g & songs.at[i, "genres"]) > 0 for i in rec_idx])


# ---------- 實驗 1：選擇 K ----------
ks = range(2, 11)
inertia, silhouette = [], []
sample = rng.choice(len(X), 10000, replace=False)  # Silhouette 計算量大，抽樣 10000 首
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(X[sample], model.labels_[sample]))
exp1 = pd.DataFrame({"K": list(ks), "inertia": inertia, "silhouette": silhouette})
exp1.to_csv("results/exp1_choose_k.csv", index=False)
print("\n[實驗 1] 選擇 K\n", exp1.round(3).to_string(index=False))

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(exp1["K"], exp1["inertia"], "o-", color="#1f77b4", label="Inertia (SSE)")
ax1.set_xlabel("Number of clusters (K)")
ax1.set_ylabel("Inertia (SSE)", color="#1f77b4")
ax2 = ax1.twinx()
ax2.plot(exp1["K"], exp1["silhouette"], "s--", color="#d62728", label="Silhouette Score")
ax2.set_ylabel("Silhouette Score", color="#d62728")
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
plt.title("Elbow Method and Silhouette Score")
fig.tight_layout()
fig.savefig("results/fig_choose_k.png", dpi=150)
plt.close(fig)

# ---------- 實驗 2：速度 vs 準確度 ----------
def evaluate(recommend):
    """對所有測試歌曲執行推薦，回傳平均 Genre Precision、平均查詢時間與推薦結果。"""
    precisions, times, results = [], [], []
    for q in queries:
        start = time.perf_counter()
        idx, _ = recommend(q)
        times.append(time.perf_counter() - start)
        precisions.append(genre_precision(q, idx))
        results.append(set(idx))
    return np.mean(precisions), np.mean(times) * 1000, results

exact_p, exact_t, exact_res = evaluate(lambda q: recommend_exact(X, q, TOP_N))
rows = [{"method": "Baseline（全量搜尋）", "K": "-", "candidates": len(X),
         "genre_precision": exact_p, "recall_vs_exact": 1.0, "query_ms": exact_t}]

for k in [3, 5, 10, 20, 50]:
    km = KMeansRecommender(X, k)
    p, t, res = evaluate(lambda q: km.recommend(q, TOP_N))
    recall = np.mean([len(a & b) / TOP_N for a, b in zip(res, exact_res)])
    cand = np.mean([len(km.members[km.labels[q]]) for q in queries])
    rows.append({"method": "K-means", "K": k, "candidates": cand,
                 "genre_precision": p, "recall_vs_exact": recall, "query_ms": t})

exp2 = pd.DataFrame(rows)
exp2.to_csv("results/exp2_speed_vs_accuracy.csv", index=False)
print("\n[實驗 2] 速度 vs 準確度\n", exp2.round(3).to_string(index=False))

km_rows = exp2[exp2["method"] == "K-means"]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(km_rows["query_ms"], km_rows["recall_vs_exact"], "o-", color="#1f77b4", label="K-means")
for _, r in km_rows.iterrows():
    ax.annotate(f"K={r['K']}", (r["query_ms"], r["recall_vs_exact"]),
                textcoords="offset points", xytext=(6, 4))
ax.scatter([exact_t], [1.0], color="#d62728", marker="*", s=200, zorder=3, label="Baseline (all songs)")
ax.set_xlabel("Average query time (ms)")
ax.set_ylabel(f"Recall@{TOP_N} vs. Baseline")
ax.set_title("Speed vs. Accuracy")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("results/fig_speed_vs_accuracy.png", dpi=150)
plt.close(fig)

# ---------- 實驗 3：特徵消融 ----------
def precision_with(features):
    Xf = scale_features(songs, features)
    return np.mean([genre_precision(q, recommend_exact(Xf, q, TOP_N)[0]) for q in queries])

random_p = np.mean([
    genre_precision(q, rng.choice(len(songs), TOP_N, replace=False)) for q in queries
])
rows = [{"setting": "隨機推薦（對照）", "genre_precision": random_p},
        {"setting": "4 項特徵（原始）", "genre_precision": precision_with(FEATURES_4)},
        {"setting": "8 項特徵", "genre_precision": precision_with(FEATURES_8)}]
for f in FEATURES_8:
    rows.append({"setting": f"8 項特徵 − {f}",
                 "genre_precision": precision_with([x for x in FEATURES_8 if x != f])})
exp3 = pd.DataFrame(rows)
exp3.to_csv("results/exp3_ablation.csv", index=False)
print("\n[實驗 3] 特徵消融（Baseline 全量搜尋）\n", exp3.round(3).to_string(index=False))

# ---------- 實驗 4：範例歌曲 ----------
km3 = KMeansRecommender(X, 3)
rows = []
for title, artist in [("Can't Help Falling in Love", "Kina Grannis"), ("Someone You Loved", "Lewis Capaldi")]:
    t = find_song(songs, title, artist)
    if t is None:
        print(f"找不到：{title} — {artist}")
        continue
    for name, (idx, dist) in [("Baseline", recommend_exact(X, t, TOP_N)), ("K-means (K=3)", km3.recommend(t, TOP_N))]:
        for rank, (i, d) in enumerate(zip(idx, dist), start=1):
            rows.append({"query": f"{title} — {artist}", "method": name, "rank": rank,
                         "title": songs.at[i, "title"], "artist": songs.at[i, "artist"],
                         "distance": round(d, 3)})
exp4 = pd.DataFrame(rows)
exp4.to_csv("results/exp4_examples.csv", index=False)
print("\n[實驗 4] 範例\n", exp4.to_string(index=False))
