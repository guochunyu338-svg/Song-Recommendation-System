import numpy as np
import pandas as pd

from recommender import FEATURES_4, FEATURES_8, find_song, load_songs, recommend_exact, scale_features

TOP_N = 3
TEST_SONGS = [  
    ("Someone You Loved", "Lewis Capaldi"),
    ("Can't Help Falling in Love", "Kina Grannis"),
]

songs = load_songs("songs.csv")
settings = {"4 features": scale_features(songs, FEATURES_4),
            "8 features": scale_features(songs, FEATURES_8)}
rng = np.random.default_rng(0)

sheet, key = [], []
for title, artist in TEST_SONGS:
    target = find_song(songs, title, artist)
    if target is None:
        print(f"找不到：{title} — {artist}")
        continue
    source = {}
    for name, X in settings.items():
        idx, _ = recommend_exact(X, target, TOP_N)
        for i in idx:
            source[i] = "both" if i in source else name
    items = list(source.items())
    rng.shuffle(items)
    for no, (i, name) in enumerate(items, start=1):
        row = {"test_song": f"{title} — {artist}", "no": no,
               "title": songs.at[i, "title"], "artist": songs.at[i, "artist"]}
        sheet.append({**row, "score (0-5)": ""})
        key.append({**row, "setting": name})

pd.DataFrame(sheet).to_csv("results/survey_sheet.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(key).to_csv("results/survey_answer_key.csv", index=False, encoding="utf-8-sig")
print("已產生 results/survey_sheet.csv 與 results/survey_answer_key.csv")
