"""產生使用者盲測問卷：比較「4 項特徵」與「8 項特徵」的推薦結果。

對每首測試歌曲，兩種設定各推薦 TOP_N 首，混合後隨機排列，
受試者看不到每首歌來自哪一種設定（盲測）。
- results/survey_sheet.csv：給受試者評分用（score 欄位留空）
- results/survey_answer_key.csv：對照表，評分完成後才打開
"""
import numpy as np
import pandas as pd

from recommender import FEATURES_4, FEATURES_8, find_song, load_songs, recommend_exact, scale_features

TOP_N = 3
TEST_SONGS = [  # 請改成受試者熟悉、且資料集中有的歌曲
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
    # 兩種設定推薦到同一首歌時只列一次，避免重複出現洩漏來源
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
