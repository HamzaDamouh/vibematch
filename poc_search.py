import time
import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Configuration
NUM_MOVIES = 100
QUERY = "sad robot movie"
DATA_PATH = "data/movies.csv"
MODEL_NAME = "all-MiniLM-L6-v2"



def main():

    start = time.time()
    # 1. Load data 
    print(f"loading data from {DATA_PATH}")
    df = pl.read_csv(DATA_PATH, columns=["movieId", "title", "genres"])

    # Semantic summary
    df = df.with_columns(
        (pl.col("title") + " — " + pl.col("genres").str.replace_all(r"\|", ", "))
        .alias("summary")
    )
    
    # 2. Generate embeddings
    print(f"Encoding first {NUM_MOVIES} movies (does take some time)")

    model = SentenceTransformer(MODEL_NAME)
    summaries = df["summary"].head(NUM_MOVIES).to_list()
    embeddings = model.encode(summaries, show_progress_bar=True)

    # FAISS expects float32
    embeddings = np.array(embeddings, dtype="float32")

    print(f"Encoded {NUM_MOVIES} movies in {time.time() - start:.2f}s")
   
    # 3. Build FAISS index
    dimension = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)



    # 4. Search 
    print(f'Searching for "{QUERY}" …')
    query_embedding = model.encode([QUERY]).astype("float32")
    distances, indices = index.search(query_embedding, k=5)
    
    print("-" * 50)

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        idx = int(idx)
        movie = df.row(idx, named=True)
        print(f"{rank}. {movie['title']} ({movie['genres']})")

    print("-" * 50)




if __name__ == "__main__":
    main()
