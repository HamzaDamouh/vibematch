import pickle
import time
from pathlib import Path



import faiss
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Config
ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "artifacts" / "movies_clean.parquet"
INDEX_PATH = ROOT / "artifacts" / "vibematch.index"
MAPPING_PATH = ROOT / "artifacts" / "movie_mapping.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
BATCH_SIZE = 256  



def build_descriptions(df: pl.DataFrame) -> list[str]:
    """Helper Function: Fallback to title + genres if description column is missing"""
    
    if "description" in df.columns:
        descriptions = df["description"].to_list()
    elif "overview" in df.columns:
        descriptions = df["overview"].to_list()
    else:
        print("  No 'description' column found — using title + genres")
        df_with_desc = df.with_columns(
            (pl.col("title")+ " — "+ pl.col("genres").str.replace_all(r"\|", ", ")).alias("_desc")
        )
        descriptions = df_with_desc["_desc"].to_list()

    return [d if isinstance(d, str) else "" for d in descriptions]


def encode_descriptions(
    descriptions: list[str], model: SentenceTransformer
) -> np.ndarray:
    """Encode all descriptions with a progress bar."""
    print(f"Encoding {len(descriptions):,} descriptions …")

    all_embeddings = []
    for start in tqdm(range(0, len(descriptions), BATCH_SIZE), desc="Encoding"):
        batch = descriptions[start : start + BATCH_SIZE]
        emb = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"   → Embedding matrix shape: {embeddings.shape}")
    return embeddings



def main() -> None:

    start_time = time.time()
   

    # 1. Load data
    print(f"Loading data from {PARQUET_PATH.name}...")
    df = pl.read_parquet(PARQUET_PATH)
    descriptions = build_descriptions(df)

    # 2. Encode
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(descriptions):,} descriptions...")
    all_embeddings = []
    for start in tqdm(range(0, len(descriptions), BATCH_SIZE), desc="Encoding"):
        batch = descriptions[start : start + BATCH_SIZE]
        all_embeddings.append(model.encode(batch, show_progress_bar=False))
    
    embeddings = np.vstack(all_embeddings).astype("float32")

    # 3. Build & Save FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(embeddings)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
 

    # 4. Save mapping
    print("Saving ID mappings...")
    movie_ids = df["movieId"].to_list()
    mapping = {
        "id_to_index": {mid: idx for idx, mid in enumerate(movie_ids)},
        "index_to_id": {idx: mid for idx, mid in enumerate(movie_ids)},
    }
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(mapping, f)

    # Summary 
    print("-" * 40)
    print(f"Build complete in {time.time() - start_time:.1f}s")
    print(f"Total encoded: {len(descriptions):,}")
    print(f"Index size: {INDEX_PATH.stat().st_size / (1024 * 1024):.2f} MB")
    print("-" * 40)


if __name__ == "__main__":
    main()
