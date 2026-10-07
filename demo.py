from recommender import FEATURES_4, KMeansRecommender, find_song, load_songs, recommend_exact, scale_features

K = 3
TOP_N = 5

songs = load_songs("songs.csv")
X = scale_features(songs, FEATURES_4)
km = KMeansRecommender(X, k=K)

title = input("Enter the title of the song: ")
artist = input("Enter the artist of the song: ")
target = find_song(songs, title, artist)

if target is None:
    print("Song not found.")
else:
    for name, (idx, dist) in [
        ("Baseline (all songs)", recommend_exact(X, target, TOP_N)),
        (f"K-means (K = {K}, same cluster only)", km.recommend(target, TOP_N)),
    ]:
        print(f"\n{name}:")
        for rank, (i, d) in enumerate(zip(idx, dist), start=1):
            print(f"{rank}. {songs.at[i, 'title']} - {songs.at[i, 'artist']} (distance: {d:.3f})")
