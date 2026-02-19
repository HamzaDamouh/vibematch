import time
from pathlib import Path

from training.preprocess import main as run_preprocess
from training.build_vectors import main as run_build_vectors
from training.train_svd import main as run_train_svd


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = {
    "movies_clean.parquet": ROOT / "artifacts" / "movies_clean.parquet",
    "vibematch.index":      ROOT / "artifacts" / "vibematch.index",
    "movie_mapping.pkl":    ROOT / "artifacts" / "movie_mapping.pkl",
    "svd_model.pkl":        ROOT / "artifacts" / "svd_model.pkl",
}


def verify_artifacts() -> bool:
    """helper: Verify all artifacts exist"""
    
    all_Aok = True
    
    for name, path in ARTIFACTS.items():
        if path.exists():
            size = path.stat().st_size
            size_str = f"{size / (1024 * 1024):.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
            print(f" [Found] {name:<25} {size_str}")
        else:
            print(f" [MISSING] {name:<25} <- The AI definitely deleted this.")
            all_Aok = False
            
    return all_Aok


def main() -> None:

    print("Starting VibeMatch Master Pipeline...")
    pipeline_start = time.time()

    steps = [
        ("Cleaning Data", run_preprocess),
        ("Building Vectors", run_build_vectors),
        ("Training Collaborative Filter", run_train_svd),
    ]

    for step_name, step_fn in steps:
        print(f"\n--- Running Step: {step_name} ---")
        step_fn()

    # Final sanity check
    all_Aok = verify_artifacts()

    total_time = time.time() - pipeline_start
    print("\n" + "*" * 50)
    if all_Aok:
        print(f"Pipeline finished successfully in {total_time:.1f}s!")
    else:
        print(f"Pipeline FAILED after {total_time:.1f}s. Check the logs above.")
    print("*" * 50)


if __name__ == "__main__":
    main()
