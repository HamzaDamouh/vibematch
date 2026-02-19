from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.engine import HybridSearchEngine


# Global engine instance
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads heavy ML artifacts into memory on startup."""
    global engine
    engine = HybridSearchEngine(
        index_path="artifacts/vibematch.index",
        svd_path="artifacts/svd_model.pkl",
        movies_path="artifacts/movies_clean.parquet",
        mapping_path="artifacts/movie_mapping.pkl",
    )
    yield
    # Clean up resources on shutdown if needed
    engine = None

app = FastAPI(title="VibeMatch API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
def search_movies(request: SearchRequest):
    results = engine.search(request.query, request.user_id, request.k)
    return {"query": request.query, "results": results}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "vibematch"}