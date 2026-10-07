"""歌曲推薦系統的共用模組：讀取資料、前處理、兩種推薦方法。

兩種方法使用「相同」的特徵、標準化與距離，唯一差別是候選範圍：
- 全量搜尋（Baseline）：與所有歌曲比較
- K-means 搜尋：只與目標歌曲同一群的歌曲比較
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES_4 = ["bpm", "energy", "danceability", "valence"]
FEATURES_8 = FEATURES_4 + ["acousticness", "instrumentalness", "loudness", "speechiness"]


def load_songs(path="songs.csv"):
    """讀取資料並清理。

    1. 排除歌名、歌手或任一特徵缺失的資料
    2. 同一首歌（歌名＋歌手相同）在資料集中可能因屬於多個曲風而重複出現，
       只保留一筆，並把它所有的曲風合併成一個集合（genres），供評估使用
    """
    songs = pd.read_csv(path)
    songs = songs.dropna(subset=["title", "artist"] + FEATURES_8)

    songs["key_title"] = songs["title"].str.lower().str.strip()
    songs["key_artist"] = songs["artist"].str.lower().str.strip()

    genres = songs.groupby(["key_title", "key_artist"])["genre"].agg(set)
    songs = songs.drop_duplicates(subset=["key_title", "key_artist"]).reset_index(drop=True)
    songs["genres"] = [genres[(t, a)] for t, a in zip(songs["key_title"], songs["key_artist"])]
    return songs


def scale_features(songs, features):
    """以 StandardScaler 將各特徵轉為平均 0、標準差 1。"""
    return StandardScaler().fit_transform(songs[features])


def find_song(songs, title, artist):
    """以歌名＋歌手（忽略大小寫與前後空白）找出目標歌曲的列號，找不到回傳 None。"""
    match = songs.index[
        (songs["key_title"] == title.lower().strip())
        & (songs["key_artist"] == artist.lower().strip())
    ]
    return match[0] if len(match) > 0 else None


def top_n(X, target_idx, candidate_idx, n):
    """在候選歌曲中，找出與目標歌曲 Euclidean distance 最小的 n 首（排除目標本身）。"""
    candidate_idx = candidate_idx[candidate_idx != target_idx]
    dist = np.sqrt(((X[candidate_idx] - X[target_idx]) ** 2).sum(axis=1))
    order = np.argsort(dist)[:n]
    return candidate_idx[order], dist[order]


def recommend_exact(X, target_idx, n=5):
    """全量搜尋（Baseline）：與所有歌曲比較。"""
    return top_n(X, target_idx, np.arange(len(X)), n)


class KMeansRecommender:
    """K-means 搜尋：先分群，推薦時只在目標歌曲所屬的群內比較。"""

    def __init__(self, X, k, seed=42):
        self.X = X
        self.labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        # 事先記錄每一群包含哪些歌曲，推薦時直接取用
        self.members = {c: np.where(self.labels == c)[0] for c in range(k)}

    def recommend(self, target_idx, n=5):
        cluster = self.labels[target_idx]
        return top_n(self.X, target_idx, self.members[cluster], n)
