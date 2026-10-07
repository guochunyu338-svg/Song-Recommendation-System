import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES_4 = ["bpm", "energy", "danceability", "valence"]
FEATURES_8 = FEATURES_4 + ["acousticness", "instrumentalness", "loudness", "speechiness"]


def load_songs(path="songs.csv"):
    songs = pd.read_csv(path)
    songs = songs.dropna(subset=["title", "artist"] + FEATURES_8)

    songs["key_title"] = songs["title"].str.lower().str.strip()
    songs["key_artist"] = songs["artist"].str.lower().str.strip()

    genres = songs.groupby(["key_title", "key_artist"])["genre"].agg(set)
    songs = songs.drop_duplicates(subset=["key_title", "key_artist"]).reset_index(drop=True)
    songs["genres"] = [genres[(t, a)] for t, a in zip(songs["key_title"], songs["key_artist"])]
    return songs


def scale_features(songs, features):
    return StandardScaler().fit_transform(songs[features])


def find_song(songs, title, artist):
    match = songs.index[
        (songs["key_title"] == title.lower().strip())
        & (songs["key_artist"] == artist.lower().strip())
    ]
    return match[0] if len(match) > 0 else None


def top_n(X, target_idx, candidate_idx, n):
    candidate_idx = candidate_idx[candidate_idx != target_idx]
    dist = np.sqrt(((X[candidate_idx] - X[target_idx]) ** 2).sum(axis=1))
    order = np.argsort(dist)[:n]
    return candidate_idx[order], dist[order]


def recommend_exact(X, target_idx, n=5):
    return top_n(X, target_idx, np.arange(len(X)), n)


class KMeansRecommender:

    def __init__(self, X, k, seed=42):
        self.X = X
        self.labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        self.members = {c: np.where(self.labels == c)[0] for c in range(k)}

    def recommend(self, target_idx, n=5):
        cluster = self.labels[target_idx]
        return top_n(self.X, target_idx, self.members[cluster], n)
