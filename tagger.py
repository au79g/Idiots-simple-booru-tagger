"""
Danbooru Image Tagger for LoRA Training
========================================
Uses the WD14 tagger model (SmilingWolf/wd-swinv2-tagger-v3 by default) to
generate Danbooru-compatible tags for every image in a folder, then saves a
matching .txt file alongside each image.

If no folder is given on the command line, a folder-selection popup will
appear so you can just click and choose — no need to edit any files.

Automatically falls back to CPU if CUDA / DirectML is unavailable.

Requirements (install once):
    pip install onnxruntime huggingface_hub pillow numpy pandas tqdm

If you have a CUDA GPU and want to use it:
    pip install onnxruntime-gpu
    (keep onnxruntime installed too – it will be used as the CPU fallback)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# Available WD models on HuggingFace – pick whichever you prefer.
AVAILABLE_MODELS = {
    "wd-swinv2-v3":     "SmilingWolf/wd-swinv2-tagger-v3",
    "wd-convnext-v3":   "SmilingWolf/wd-convnext-tagger-v3",
    "wd-vit-v3":        "SmilingWolf/wd-vit-tagger-v3",
    "wd-vit-large-v3":  "SmilingWolf/wd-vit-large-tagger-v3",
    "wd-eva02-large-v3":"SmilingWolf/wd-eva02-large-tagger-v3",
    # older but still decent:
    "wd-v1-4-convnext": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "wd-v1-4-swinv2":   "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
}

DEFAULT_MODEL     = "wd-swinv2-v3"
DEFAULT_THRESHOLD = 0.35   # confidence threshold for general tags
RATING_THRESHOLD  = 0.5    # confidence threshold for rating tags
CHARACTER_THRESHOLD = 0.85  # higher threshold for character tags (more precise)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Folder picker (used when no folder is passed on the command line)
# ---------------------------------------------------------------------------

def prompt_for_folder() -> Path | None:
    """
    Show a native 'choose folder' dialog and return the selected path.
    Returns None if the user cancels or if a GUI isn't available (e.g. a
    headless server) — in that case the caller should fall back to asking
    on the command line.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    try:
        root = tk.Tk()
        root.withdraw()          # hide the empty main window
        root.attributes("-topmost", True)  # bring the dialog to the front
        folder = filedialog.askdirectory(
            title="Select the folder of images to tag"
        )
        root.destroy()
    except Exception:
        return None

    return Path(folder) if folder else None


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------

def download_model(repo_id: str, cache_dir: Path) -> tuple[Path, Path]:
    """
    Download the ONNX model and tag CSV from HuggingFace if not already cached.
    Returns (model_path, tags_csv_path).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading / verifying model: {repo_id}")

    model_path = Path(hf_hub_download(
        repo_id=repo_id,
        filename="model.onnx",
        cache_dir=str(cache_dir),
    ))

    tags_path = Path(hf_hub_download(
        repo_id=repo_id,
        filename="selected_tags.csv",
        cache_dir=str(cache_dir),
    ))

    return model_path, tags_path


# ---------------------------------------------------------------------------
# ONNX session with automatic provider fallback
# ---------------------------------------------------------------------------

def create_onnx_session(model_path: Path, force_cpu: bool = False):
    """
    Create an ONNX Runtime inference session.
    Provider priority: CUDAExecutionProvider → CPUExecutionProvider
    Falls back gracefully to CPU if CUDA isn't available or throws an error.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    available = ort.get_available_providers()

    if force_cpu:
        providers = ["CPUExecutionProvider"]
        print("  [Provider] Forced CPU mode.")
    elif "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("  [Provider] CUDA GPU detected – will try GPU first, CPU as fallback.")
    elif "DmlExecutionProvider" in available:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        print("  [Provider] DirectML (AMD/Intel GPU) detected.")
    else:
        providers = ["CPUExecutionProvider"]
        print("  [Provider] No GPU provider found – using CPU.")

    # Try creating the session; fall back to CPU-only if GPU init fails
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
        active = session.get_providers()
        print(f"  [Provider] Active provider: {active[0]}")
        return session
    except Exception as gpu_err:
        print(f"  [Warning] GPU session failed ({gpu_err}), falling back to CPU.")
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        print("  [Provider] Active provider: CPUExecutionProvider")
        return session


# ---------------------------------------------------------------------------
# Tag loading
# ---------------------------------------------------------------------------

def load_tags(tags_csv: Path) -> tuple[list, list, list]:
    """
    Parse selected_tags.csv and return three lists:
    (tag_names, general_indexes, character_indexes)
    The CSV has columns: tag_id, name, category, count
      category 0 = general, category 4 = character, category 9 = rating
    """
    tag_names        = []
    general_indexes  = []
    character_indexes = []
    rating_indexes   = []

    with open(tags_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tag_names.append(row["name"])
            cat = int(row.get("category", 0))
            if cat == 9:
                rating_indexes.append(i)
            elif cat == 4:
                character_indexes.append(i)
            else:
                general_indexes.append(i)

    return tag_names, general_indexes, character_indexes, rating_indexes


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: Path, target_size: int = 448) -> np.ndarray:
    """
    Load and preprocess an image to the format expected by WD tagger models:
    - Resize with padding to a square (target_size × target_size)
    - RGB → BGR channel order
    - Float32, values 0-255 (NOT normalised to 0-1)
    - Shape: (1, H, W, C)
    """
    img = Image.open(image_path).convert("RGBA")

    # Composite onto white background (handles transparency)
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    canvas.paste(img, mask=img.split()[3])
    img = canvas.convert("RGB")

    # Resize keeping aspect ratio, pad to square
    img.thumbnail((target_size, target_size), Image.LANCZOS)
    padded = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset = ((target_size - img.width) // 2, (target_size - img.height) // 2)
    padded.paste(img, offset)

    arr = np.array(padded, dtype=np.float32)
    arr = arr[:, :, ::-1]          # RGB → BGR
    arr = np.expand_dims(arr, 0)   # Add batch dimension
    return arr


# ---------------------------------------------------------------------------
# Tagging a single image
# ---------------------------------------------------------------------------

def tag_image(
    session,
    image_path: Path,
    tag_names: list,
    general_indexes: list,
    character_indexes: list,
    rating_indexes: list,
    general_threshold: float,
    character_threshold: float,
    include_ratings: bool,
) -> list[str]:
    """
    Run inference on a single image and return a list of Danbooru tags.
    """
    input_name = session.get_inputs()[0].name
    arr = preprocess_image(image_path)

    preds = session.run(None, {input_name: arr})[0][0]  # shape: (num_tags,)

    result_tags = []

    # Rating tags (optional)
    if include_ratings:
        rating_preds = {tag_names[i]: preds[i] for i in rating_indexes}
        best_rating = max(rating_preds, key=rating_preds.get)
        result_tags.append(best_rating)

    # Character tags (higher threshold → more precise)
    for i in character_indexes:
        if preds[i] >= character_threshold:
            result_tags.append(tag_names[i])

    # General tags
    for i in general_indexes:
        if preds[i] >= general_threshold:
            result_tags.append(tag_names[i])

    # Replace underscores with spaces (standard Danbooru/LoRA format)
    result_tags = [t.replace("_", " ") for t in result_tags]

    return result_tags


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_folder(
    input_folder: Path,
    output_folder: Path | None,
    session,
    tag_names: list,
    general_indexes: list,
    character_indexes: list,
    rating_indexes: list,
    general_threshold: float,
    character_threshold: float,
    include_ratings: bool,
    overwrite: bool,
    recursive: bool,
):
    """Walk the input folder and tag every image found."""

    if recursive:
        image_files = [
            p for p in input_folder.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        image_files = [
            p for p in input_folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    if not image_files:
        print("No image files found in the specified folder.")
        return

    print(f"\nFound {len(image_files)} image(s). Starting tagging...\n")
    errors = []

    for img_path in tqdm(image_files, unit="img"):
        # Determine where to write the .txt file
        if output_folder:
            # Mirror subfolder structure under output_folder
            rel = img_path.relative_to(input_folder)
            txt_path = output_folder / rel.with_suffix(".txt")
            txt_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            txt_path = img_path.with_suffix(".txt")

        if txt_path.exists() and not overwrite:
            continue  # Skip already-tagged images unless --overwrite

        try:
            tags = tag_image(
                session, img_path,
                tag_names, general_indexes, character_indexes, rating_indexes,
                general_threshold, character_threshold, include_ratings,
            )
            txt_path.write_text(", ".join(tags), encoding="utf-8")
        except Exception as e:
            errors.append((img_path, str(e)))
            tqdm.write(f"  [Error] {img_path.name}: {e}")

    print(f"\nDone! Tagged {len(image_files) - len(errors)} image(s).")
    if errors:
        print(f"{len(errors)} image(s) failed – see errors above.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Danbooru tagger for LoRA training datasets (WD14 / WD3 models)"
    )
    p.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Folder containing images to tag. If omitted, a folder-"
             "selection dialog will pop up.",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Optional output folder for .txt files. "
             "Defaults to same folder as each image.",
    )
    p.add_argument(
        "--model", "-m",
        choices=list(AVAILABLE_MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Which WD tagger model to use. Default: {DEFAULT_MODEL}",
    )
    p.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold for general tags (0–1). Default: {DEFAULT_THRESHOLD}",
    )
    p.add_argument(
        "--character-threshold",
        type=float,
        default=CHARACTER_THRESHOLD,
        help=f"Confidence threshold for character tags. Default: {CHARACTER_THRESHOLD}",
    )
    p.add_argument(
        "--include-ratings",
        action="store_true",
        help="Include the image rating tag (general/sensitive/explicit/questionable).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode even if a GPU is available.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-tag images that already have a .txt file.",
    )
    p.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search for images recursively in subdirectories.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "wd_tagger",
        help="Directory to cache downloaded models.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # No folder given on the command line -> pop up a folder picker dialog.
    if args.input is None:
        print("No input folder given on the command line — opening folder picker...")
        chosen = prompt_for_folder()
        if chosen is None:
            print("ERROR: No folder was selected. Either pick a folder in the "
                  "dialog, or run this script with a folder path, e.g.:\n"
                  "    python tagger.py \"C:\\path\\to\\images\"")
            sys.exit(1)
        args.input = chosen

    if not args.input.is_dir():
        print(f"ERROR: '{args.input}' is not a directory.")
        sys.exit(1)

    repo_id = AVAILABLE_MODELS[args.model]

    print("=" * 60)
    print("  Danbooru Tagger for LoRA Training")
    print("=" * 60)
    print(f"  Input folder  : {args.input}")
    print(f"  Output folder : {args.output or '(same as input)'}")
    print(f"  Model         : {args.model} ({repo_id})")
    print(f"  Gen threshold : {args.threshold}")
    print(f"  Char threshold: {args.character_threshold}")
    print(f"  Include rating: {args.include_ratings}")
    print(f"  Force CPU     : {args.cpu}")
    print(f"  Overwrite     : {args.overwrite}")
    print(f"  Recursive     : {args.recursive}")
    print("=" * 60 + "\n")

    # Step 1: Download model
    model_path, tags_csv = download_model(repo_id, args.cache_dir)

    # Step 2: Load tags
    print("Loading tag definitions...")
    tag_names, general_indexes, character_indexes, rating_indexes = load_tags(tags_csv)
    print(f"  Loaded {len(tag_names)} tags "
          f"({len(general_indexes)} general, "
          f"{len(character_indexes)} character, "
          f"{len(rating_indexes)} rating)\n")

    # Step 3: Create ONNX session
    print("Loading model...")
    session = create_onnx_session(model_path, force_cpu=args.cpu)
    print()

    # Step 4: Process images
    process_folder(
        input_folder=args.input,
        output_folder=args.output,
        session=session,
        tag_names=tag_names,
        general_indexes=general_indexes,
        character_indexes=character_indexes,
        rating_indexes=rating_indexes,
        general_threshold=args.threshold,
        character_threshold=args.character_threshold,
        include_ratings=args.include_ratings,
        overwrite=args.overwrite,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
