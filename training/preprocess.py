import polars as pl
from pathlib import Path

# Config
ROOT = Path(__file__).resolve().parent.parent
MOVIES_PATH = ROOT / "data" / "movies.csv"
RATINGS_PATH = ROOT / "data" / "ratings.csv"
OUTPUT_PATH = ROOT / "artifacts" / "movies_clean.parquet"

MIN_RATINGS = 5  # drop movies with fewer ratings




def clean_movie_data() ->pl.DataFrame:
    """helper function: Loads + MergeS + Clean MovieLens datasets """

    print("Loading raw CSVs...")
    movies = pl.read_csv(MOVIES_PATH)
    ratings = pl.read_csv(RATINGS_PATH)

    print("Aggregating rating statistics...")
    rating_stats = ratings.group_by("movieId").agg(
        pl.col("rating").mean().alias("avg_rating"),
        pl.col("rating").count().alias("rating_count"),
    )

    df = movies.join(rating_stats, on="movieId", how="left")
    
    df = df.with_columns(
        pl.col("avg_rating").fill_null(0.0),
        pl.col("rating_count").fill_null(0),
    )

    initial_count = len(df)
    df = df.filter(pl.col("rating_count") >= MIN_RATINGS)
    print(f"Filtered out {initial_count - len(df):,} movies with fewer than {MIN_RATINGS} ratings.")

    # Build semantic description
    df = df.with_columns(
        (pl.col("title") + " — " + pl.col("genres").str.replace_all(r"\|", ", "))
        .alias("description")
    )

    return df
    
def main() -> None:

    df = clean_movie_data()
    
    print(f"Saving {len(df):,} cleaned movies to Parquet...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_PATH)
    
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
