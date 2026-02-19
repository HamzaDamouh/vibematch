import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------
# If RMSE < 0.8 : I'm genius
# Elif > 1.5 : dataset is clearly flawed
# ---------------------------------------------------------

# Config
ROOT = Path(__file__).resolve().parent.parent
RATINGS_PATH = ROOT / "data" / "ratings.csv"
MODEL_PATH = ROOT / "artifacts" / "svd_model.pkl"

N_FACTORS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_ratings() -> pl.DataFrame:
    """helper: Load ratings and display stats"""
    df = pl.read_csv(RATINGS_PATH, columns=["userId", "movieId", "rating"])
    print(f"  {len(df):,} ratings | {df['userId'].n_unique():,} users | {df['movieId'].n_unique():,} movies")
    return df


def train_test_split(df: pl.DataFrame, test_size: float):
    """helper: train/test split (random shuffle)"""
    df = df.sample(fraction=1.0, shuffle=True, seed=RANDOM_STATE)
    split_idx = int(len(df) * (1 - test_size))
    return df[:split_idx], df[split_idx:]   


def build_sparse_matrix(df: pl.DataFrame):
    """helper: Build user-movie sparse matrix + ID mappings"""

    user_ids = df["userId"].unique().sort().to_list()
    movie_ids = df["movieId"].unique().sort().to_list()

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    rows = [user_to_idx[u] for u in df["userId"].to_list()]
    cols = [movie_to_idx[m] for m in df["movieId"].to_list()]
    vals = df["rating"].to_list()

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    )

    return matrix, user_to_idx, movie_to_idx, user_ids, movie_ids


def compute_biases(train_df: pl.DataFrame, global_mean: float):
    """helper: deviation from global mean"""

    # Isolate the haters and the fanboys
    user_bias = (
        train_df.group_by("userId")
        .agg((pl.col("rating").mean() - global_mean).alias("bias"))
    )
    item_bias = (
        train_df.group_by("movieId")
        .agg((pl.col("rating").mean() - global_mean).alias("bias"))
    )
    user_bias_dict = dict(zip(user_bias["userId"].to_list(), user_bias["bias"].to_list()))
    item_bias_dict = dict(zip(item_bias["movieId"].to_list(), item_bias["bias"].to_list()))
    return user_bias_dict, item_bias_dict


def predict(uid, mid, user_factors, item_factors, user_to_idx, movie_to_idx,
            global_mean, user_bias, item_bias): 
    """helper: Predict rating: global_mean + user_bias + item_bias + dot(u, i)"""
    bu = user_bias.get(uid, 0.0)
    bi = item_bias.get(mid, 0.0)

    if uid in user_to_idx and mid in movie_to_idx:
        u_vec = user_factors[user_to_idx[uid]]
        i_vec = item_factors[movie_to_idx[mid]]
        pred = global_mean + bu + bi + np.dot(u_vec, i_vec)
    else:
        pred = global_mean + bu + bi  # cold start fallback

    return float(np.clip(pred, 0.5, 5.0)) # standard [0.5, 5.0] range


def main() -> None:

    df = load_ratings()

    train_df, test_df = train_test_split(df, TEST_SIZE)
   
    print("Building user-movie matrix")
    matrix, user_to_idx, movie_to_idx, user_ids, movie_ids = build_sparse_matrix(train_df)

    global_mean = train_df["rating"].mean()
    user_bias, item_bias = compute_biases(train_df, global_mean)


    print(f"Training TruncatedSVD (factors={N_FACTORS}) coffee time")
    svd = TruncatedSVD(n_components=N_FACTORS, random_state=RANDOM_STATE)

    start = time.time()
    user_factors = svd.fit_transform(matrix)   # (n_users, n_factors)
    item_factors = svd.components_.T            # (n_movies, n_factors)
    train_time = time.time() - start

    print("Evaluating on test set")
    y_true, y_pred = [], []
    for row in test_df.iter_rows(named=True):
        uid, mid, rating = row["userId"], row["movieId"], row["rating"]
        pred = predict(uid, mid, user_factors, item_factors,
                       user_to_idx, movie_to_idx, global_mean, user_bias, item_bias)
        y_true.append(rating)
        y_pred.append(pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    model_data = {
        "svd": svd,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "global_mean": global_mean,
        "user_bias": user_bias,
        "item_bias": item_bias,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, MODEL_PATH)
    

    print("-" * 50)
    print(f"Algorithm Accuracy (RMSE): {rmse:.4f} stars")
    print(f"Training time:             {train_time:.1f}s")
    print(f"Model saved to:            {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB)")
    print("-" * 50)

if __name__ == "__main__":
    main()
