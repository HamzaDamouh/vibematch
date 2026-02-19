import pickle
import time
from pathlib import Path

import faiss
import joblib
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
N_CANDIDATES = 50  


class HybridSearchEngine:
    """
    Two-stage recommendaer : FAISS for semantic retrieval and SVD for collaborative filtering
    """

    def __init__(self, index_path: str, svd_path: str, movies_path: str, mapping_path: str):
        

        self.index = faiss.read_index(index_path)
        self.movies = pl.read_parquet(movies_path)

        
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)

        self.id_to_index = mapping["id_to_index"]
        self.index_to_id = mapping["index_to_id"]

        
        svd_data = joblib.load(svd_path)
        self.user_factors = svd_data["user_factors"]
        self.item_factors = svd_data["item_factors"]
        self.user_to_idx = svd_data["user_to_idx"]
        self.movie_to_idx = svd_data["movie_to_idx"]
        self.global_mean = svd_data["global_mean"]
        self.user_bias = svd_data["user_bias"]
        self.item_bias = svd_data["item_bias"]

        self.encoder = SentenceTransformer(MODEL_NAME)
        print(f"Engine initialized with {self.index.ntotal} movies")

    
    def _predict_rating(self, user_id: int, movie_id: int) -> float:
       
        bu = self.user_bias.get(user_id, 0.0)
        bi = self.item_bias.get(movie_id, 0.0)

        if user_id in self.user_to_idx and movie_id in self.movie_to_idx:
            u_vec = self.user_factors[self.user_to_idx[user_id]]
            i_vec = self.item_factors[self.movie_to_idx[movie_id]]
            pred = self.global_mean + bu + bi + np.dot(u_vec, i_vec)
        else:
            pred = self.global_mean + bu + bi

        return float(np.clip(pred, 0.5, 5.0))

    
    
    def search(self, query: str, user_id: int | None = None, k: int = 5) -> list[dict]:
       
        query_vec = self.encoder.encode([query]).astype("float32")

        n_candidates = min(N_CANDIDATES, self.index.ntotal)
        distances, indices = self.index.search(query_vec, n_candidates)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            idx = int(idx)
            movie_id = self.index_to_id[idx]

            movie_row = self.movies.filter(pl.col("movieId") == movie_id)
            if movie_row.is_empty():
                continue

            movie = movie_row.row(0, named=True)
            similarity = float(1 / (1 + dist))

            if user_id is not None:
                predicted_rating = self._predict_rating(user_id, movie_id)
            else:
                predicted_rating = movie.get("avg_rating", self.global_mean)

            # Blend: 60% semantic + 40% personalized quality
            norm_rating = (predicted_rating - 0.5) / 4.5
            blended_score = round(0.6 * similarity + 0.4 * norm_rating, 4)

            results.append({
                "movie_id": movie_id,
                "title": movie["title"],
                "genres": movie["genres"],
                "similarity_score": round(similarity, 4),
                "predicted_rating": round(predicted_rating, 2),
                "score": blended_score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]


if __name__ == "__main__":

    # Quick test local

    ROOT = Path(__file__).resolve().parent.parent
    engine = HybridSearchEngine(
        index_path=str(ROOT / "artifacts" / "vibematch.index"),
        svd_path=str(ROOT / "artifacts" / "svd_model.pkl"),
        movies_path=str(ROOT / "artifacts" / "movies_clean.parquet"),
        mapping_path=str(ROOT / "artifacts" / "movie_mapping.pkl"),
    )

    queries = ["space adventure", "romantic comedy"]

    for q in queries:
        print(f"\nSearching: '{q}'")
        start = time.time()
        results = engine.search(q, user_id=None, k=5)

        for i, r in enumerate(results, 1):
            print(f" {i}. {r['title']} ({r['genres']}) | Score: {r['score']} | Sim: {r['similarity_score']}")
        print(f"Took: {(time.time() - start) * 1000:.0f}ms")
