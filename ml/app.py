import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResponseType(BaseModel):
    feats: List[str]
    means: List[float]
    mins: List[float]
    maxes: List[float]
    song_names: List[str]
    artists: List[List[str]]


class PlaylistFinder:
    def __init__(self, df: pd.DataFrame, df_genres: pd.DataFrame):
        self.df = df
        self.df_genres = df_genres

    @staticmethod
    def is_corrupt(value: str) -> int:
        found = bool(re.search("[0-9A-Za-z]", value))
        found = found and "remaster" not in value.lower()
        return int(not found)

    def process_data(self) -> None:
        genre_mappings: Dict[str, int] = {}

        def get_genre(value: str) -> int:
            for artist in value[1:-1].replace("'", "").split(","):
                artist_genres = self.df_genres.loc[
                    self.df_genres.artists == artist
                ].genres.values
                if artist_genres == "[]" or not artist_genres:
                    continue
                artist_genres = artist_genres[0]
                possible_candidates = artist_genres[1:-1].replace("'", "").split(",")
                last_suggestion = ""
                inserting_new_genre = True
                for candidate in possible_candidates:
                    last_suggestion = candidate.split(" ")[-1]
                    if last_suggestion not in genre_mappings:
                        continue
                    else:
                        inserting_new_genre = False
                        break
                if inserting_new_genre:
                    genre_mappings[last_suggestion] = len(genre_mappings)

                return genre_mappings[last_suggestion]
            return -1

        # Filter out elements with corrupt artists/names fields
        # along with elements with unspecified artists
        self.df["artists_corrupt"] = self.df["artists"].apply(
            lambda x: self.is_corrupt(x)
        )
        self.df["name_corrupt"] = self.df["name"].apply(lambda x: self.is_corrupt(x))
        self.df = self.df[self.df["artists_corrupt"] == 0]
        self.df = self.df[self.df["name_corrupt"] == 0]
        self.df = self.df[self.df["artists"] != "['Unspecified']"]
        # Filter out elements with popularity of 0
        self.df = self.df[self.df["popularity"] > 0]
        # Filter out elements that are "vocal only", such as speeches etc.
        self.df = self.df[self.df["speechiness"] < 0.66]
        # Filter by year (used to reduce the number of samples, because clustering is very slow otherwise)
        self.df = self.df[self.df["year"] >= 2000]
        # There are some records of songs entered twice with different values for the features, removing them
        self.df = self.df.drop_duplicates(["artists", "name"])
        self.df = self.df.sample(10000)
        # insert genres into dataframe and remove songs with no genre
        self.df["genre"] = self.df["artists"].apply(lambda x: get_genre(x))
        self.df = self.df[self.df["genre"] != -1]

    def run_clustering(self) -> pd.DataFrame:
        # create a dataframe of the features needed for training
        df_feats = self.df.drop(
            [
                "artists",
                "name",
                "artists_corrupt",
                "name_corrupt",
                "id",
                "year",
                "release_date",
                "key",
                "instrumentalness",
                "popularity",
                "energy",
            ],
            axis=1,
        )
        # normalize features, each by its own column, and again the whole bunch
        # (this yielded the best results and also ensured all values have the same distribution)
        df_scaled_feats = normalize(df_feats, axis=0)
        df_scaled_feats = normalize(df_scaled_feats)
        # Reduce dimensionality to 2 features (I also experimented with 3, but 2 seemed good enough)
        pca = PCA(n_components=2)
        df_pca_feats = pca.fit_transform(df_scaled_feats)
        # Run KMeans Clustering
        # I experimented a lot with other clustering methods (OPTICS, DBScan, Agglomerative, etc.)
        # In the end KMeans turned out to give the best results, all factors considered
        cluster = KMeans(n_clusters=120)
        cluster.fit(df_pca_feats)
        self.df["clusters"] = cluster.labels_
        df_scaled_feats = pd.DataFrame(
            df_scaled_feats, columns=df_feats.columns.tolist()
        )
        df_scaled_feats.index = df_feats.index.tolist()
        return df_scaled_feats

    def get_output_data(self, df_scaled_feats: pd.DataFrame) -> List[Dict]:
        CHECK_FEATS = ["acousticness", "danceability", "liveness", "loudness", "genre"]
        max_values = {feat: df_scaled_feats[feat].max() for feat in CHECK_FEATS}
        min_values = {feat: df_scaled_feats[feat].min() for feat in CHECK_FEATS}
        cluster_info: List[Dict] = []
        for cluster_idx, cluster in enumerate(
            np.random.choice(
                self.df["clusters"].unique(), min(len(self.df["clusters"].unique()), 50)
            )
        ):
            if len(cluster_info) > 10:
                break
            cluster_df = self.df[self.df["clusters"] == int(cluster)]
            idxs = (
                self.df[self.df["clusters"] == int(cluster)]
                .sample(min(len(cluster_df), 30))
                .index
            )
            df_feats_chunk = df_scaled_feats.loc[idxs]
            chunk_stds: List[float] = []
            chunk_means: List[float] = []

            for feat in CHECK_FEATS:
                chunk_means.append(df_feats_chunk[feat].mean())
                chunk_stds.append(df_feats_chunk[feat].std())

            chunk_stds_np = np.asarray(chunk_stds)
            chunk_means_np = np.asarray(chunk_means)

            argsorted = np.argsort(chunk_stds_np)
            chunk_stds = chunk_stds_np[argsorted]
            chunk_means = chunk_means_np[argsorted]

            chunk_stds = list(chunk_stds_np)
            chunk_means = list(chunk_means_np)

            feat_names = [CHECK_FEATS[i] for i in argsorted[:3]]
            song_names = cluster_df.loc[idxs].name.values.tolist()
            artist_names = (
                cluster_df.loc[idxs]
                .artists.apply(
                    lambda x: x[1:-1].replace("'", "").replace(", ", ",").split(",")
                )
                .values.tolist()
            )
            cluster_info.append(
                {
                    "feats": feat_names,
                    "means": chunk_means[:3],
                    "mins": [min_values[feat] for feat in feat_names],
                    "maxes": [max_values[feat] for feat in feat_names],
                    "song_names": song_names,
                    "artists": artist_names,
                }
            )
        return cluster_info

    def __call__(self) -> List[Dict]:
        self.process_data()
        df_scaled_feats = self.run_clustering()
        return self.get_output_data(df_scaled_feats)


DATA_DIR = Path("data")


@app.get("/playlists")
async def get_playlists() -> List[ResponseType]:
    df = pd.read_csv(DATA_DIR / "data.csv")
    df_genres = pd.read_csv(DATA_DIR / "data_w_genres.csv")
    service = PlaylistFinder(df, df_genres)
    res_list = service()
    res = [ResponseType(**res_dict) for res_dict in res_list]
    return res
