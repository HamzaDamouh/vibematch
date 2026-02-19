from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr
from app.engine import HybridSearchEngine


class SearchRequest(BaseModel):
    query: str
    user_id: int | None = None
    k: int = 5


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
    engine = None

app = FastAPI(title="VibeMatch API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ───────────────────────────────────────

@app.post("/search")
def search_movies(request: SearchRequest):
    results = engine.search(request.query, request.user_id, request.k)
    return {"query": request.query, "results": results}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "vibematch"}


# ── Gradio Interface ───────────────────────────────────

def gradio_search(query: str) -> str:
    """Search and format results as markdown."""
    if not query.strip():
        return "*Type a vibe to get started...*"

    results = engine.search(query, user_id=None, k=5)

    if not results:
        return "No movies found. Try a different description!"

    lines = [f"### Results for *\"{query}\"*\n"]
    for i, r in enumerate(results, 1):
        stars = "⭐" * round(r["predicted_rating"])
        lines.append(f"**{i}. {r['title']}**")
        lines.append(f"  {r['genres'].replace('|', ' · ')}")
        lines.append(f"  {stars} ({r['predicted_rating']:.1f})  ·  Similarity: {r['similarity_score']:.2%}")
        lines.append("")

    return "\n".join(lines)


interface = gr.Interface(
    fn=gradio_search,
    inputs=gr.Textbox(
        label="Describe the vibe you want",
        placeholder="e.g., sad robot movie",
        lines=2,
    ),
    outputs=gr.Markdown(label="🎬 Recommendations"),
    examples=[
        "sad robot movie",
        "action thriller in space",
        "feel-good romance",
        "mind-bending sci-fi",
        "emotional drama",
    ],
    title="🎬 VibeMatch — Find Movies by Vibe",
    description="Semantic search powered by AI embeddings + collaborative filtering",
)

app = gr.mount_gradio_app(app, interface, path="/")