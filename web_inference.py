""" IMPORTS """
import csv
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as TF

# Import reusable model/config pieces from existing project files
from brine_shrimp_train_and_infer import (
    ROOT,
    BEST_MODEL_PATH,
    SAVE_ROOT,
    get_detector,
    IMG_SUFFIXES,
    is_image_ok,
)

""" CONFIGURATION """
CONF_THRESH = 0.85

# Website-specific folders
UPLOADS_DIR = ROOT / "uploads"
WEB_RESULTS_DIR = ROOT / "web_results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
WEB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Also create static output folder so Flask can display result images easily
STATIC_OUTPUTS_DIR = ROOT / "static" / "outputs"
STATIC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


""" FILE VALIDATION """
def allowed_file(filename: str) -> bool:
    """Return True if the uploaded filename has an allowed image extension."""
    if not filename or "." not in filename:
        return False
    suffix = Path(filename).suffix
    return suffix in IMG_SUFFIXES


def validate_uploaded_files(files: List) -> Tuple[bool, str]:
    """
    Validate uploaded files from Flask request.files.getlist("images").

    Rules:
    - Must upload between 1 and 10 files
    - All files must have a filename
    - All files must be valid supported image types
    """
    if not files or len(files) == 0:
        return False, "Please upload at least 1 image."

    if len(files) > 10:
        return False, "You can upload a maximum of 10 images at a time."

    for file in files:
        if not file or not file.filename:
            return False, "One of the uploaded files is missing a filename."
        if not allowed_file(file.filename):
            allowed_types = ", ".join(sorted(s.lower() for s in IMG_SUFFIXES if s.startswith(".")))
            return False, f"Unsupported file type for '{file.filename}'. Allowed types: {allowed_types}"

    return True, ""


""" MODEL LOADING """
def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(device: torch.device = None):
    """
    Load the saved trained Faster R-CNN model and return (model, device).

    This should be called once when the Flask app starts, not for every request.
    """
    if device is None:
        device = get_device()

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {BEST_MODEL_PATH}. "
            "Train the model first or place brineshrimp_best_model.pth in shrimp_runs/."
        )

    model = get_detector(num_classes=2).to(device)

    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[web_inference] Loaded trained model from {BEST_MODEL_PATH}")
    return model, device


""" SAVE HELPERS """
def create_batch_id() -> str:
    """Create a unique batch id for a website upload session."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_id}"


def safe_filename(filename: str) -> str:
    """Return a simplified safe filename."""
    path = Path(filename)
    stem = path.stem.replace(" ", "_")
    suffix = path.suffix.lower()
    return f"{stem}{suffix}"


def save_uploaded_file(file_storage, batch_id: str) -> Path:
    """
    Save one uploaded file into uploads/ using a unique batch-prefixed filename.
    Returns the saved file path.
    """
    original_name = safe_filename(file_storage.filename)
    saved_name = f"{batch_id}_{original_name}"
    save_path = UPLOADS_DIR / saved_name
    file_storage.save(save_path)
    return save_path


def _build_output_paths(image_path: Path, batch_id: str) -> Tuple[Path, Path]:
    """
    Build:
    - website result image path in web_results/
    - displayable result image path in static/outputs/
    """
    segmented_name = f"{batch_id}_segmented_{image_path.stem}.png"
    web_result_path = WEB_RESULTS_DIR / segmented_name
    static_result_path = STATIC_OUTPUTS_DIR / segmented_name
    return web_result_path, static_result_path


""" INFERENCE HELPERS """
@torch.no_grad()
def predict_single_image(
    model,
    device: torch.device,
    image_path: Path,
    batch_id: str,
    conf_thresh: float = CONF_THRESH,
) -> Dict:
    """
    Run inference on one uploaded image, save an annotated output image,
    and return a dictionary of result metadata.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Uploaded image not found: {image_path}")

    if not is_image_ok(image_path):
        raise ValueError(f"Unreadable image file: {image_path.name}")

    # Open and prepare image
    img = Image.open(image_path).convert("RGB")
    inp = TF.to_tensor(img).unsqueeze(0).to(device)

    # Run model
    output = model(inp)[0]

    # Filter detections by confidence threshold
    scores = output["scores"].detach().cpu().numpy()
    keep = scores >= conf_thresh

    boxes = output["boxes"].detach().cpu().numpy()[keep]
    kept_scores = scores[keep]
    count = int(keep.sum())

    # Save annotated image
    annotated_img = img.copy()
    draw = ImageDraw.Draw(annotated_img)

    for (xmin, ymin, xmax, ymax), score in zip(boxes, kept_scores):
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0), width=3)
        draw.text((xmin, max(0, ymin - 14)), f"{score:.2f}", fill=(0, 255, 0))

    web_result_path, static_result_path = _build_output_paths(image_path, batch_id)

    # Save in both places so you keep a website-specific archive and a displayable static file
    annotated_img.save(web_result_path)
    annotated_img.save(static_result_path)

    return {
        "batch_id": batch_id,
        "original_filename": image_path.name,
        "uploaded_image_path": str(image_path),
        "uploaded_image_name": image_path.name,
        "count": count,
        "num_detections": count,
        "annotated_image_path": str(web_result_path),
        "annotated_image_name": static_result_path.name,
        "annotated_image_url": f"outputs/{static_result_path.name}",
        "confidence_threshold": conf_thresh,
    }


@torch.no_grad()
def predict_batch_images(
    model,
    device: torch.device,
    image_paths: List[Path],
    batch_id: str,
    conf_thresh: float = CONF_THRESH,
) -> List[Dict]:
    """
    Run inference on a batch of 1-10 saved uploaded images.
    Returns a list of result dictionaries.
    """
    results = []

    for image_path in image_paths:
        try:
            result = predict_single_image(
                model=model,
                device=device,
                image_path=image_path,
                batch_id=batch_id,
                conf_thresh=conf_thresh,
            )
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "batch_id": batch_id,
                    "original_filename": Path(image_path).name,
                    "uploaded_image_path": str(image_path),
                    "uploaded_image_name": Path(image_path).name,
                    "count": 0,
                    "num_detections": 0,
                    "annotated_image_path": "",
                    "annotated_image_name": "",
                    "annotated_image_url": "",
                    "confidence_threshold": conf_thresh,
                    "error": str(e),
                }
            )

    return results


""" CSV OUTPUT """
def write_batch_csv(results: List[Dict], batch_id: str) -> Path:
    """
    Write a CSV summary for a website upload batch.
    Returns the CSV file path.
    """
    csv_path = WEB_RESULTS_DIR / f"{batch_id}_brine_shrimp_counts.csv"

    total = sum(item.get("count", 0) for item in results if not item.get("error"))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Count", "Error"])

        for item in results:
            writer.writerow(
                [
                    item.get("original_filename", ""),
                    item.get("count", 0),
                    item.get("error", ""),
                ]
            )

        writer.writerow([])
        writer.writerow(["Total Shrimp Detected", total, ""])

    return csv_path


""" FULL WEBSITE BATCH WORKFLOW """
def process_uploaded_files(
    files: List,
    model,
    device: torch.device,
    conf_thresh: float = CONF_THRESH,
) -> Dict:
    """
    Full website workflow for 1-10 uploaded images:
    - validate files
    - save uploads
    - run inference on each image
    - save CSV summary
    - return all website-ready metadata
    """
    valid, message = validate_uploaded_files(files)
    if not valid:
        raise ValueError(message)

    batch_id = create_batch_id()
    saved_paths = []

    for file in files:
        saved_path = save_uploaded_file(file, batch_id)
        saved_paths.append(saved_path)

    results = predict_batch_images(
        model=model,
        device=device,
        image_paths=saved_paths,
        batch_id=batch_id,
        conf_thresh=conf_thresh,
    )

    csv_path = write_batch_csv(results, batch_id)
    total_count = sum(item.get("count", 0) for item in results if not item.get("error"))

    return {
        "batch_id": batch_id,
        "results": results,
        "total_count": total_count,
        "num_uploaded": len(saved_paths),
        "csv_path": str(csv_path),
        "csv_name": csv_path.name,
        "confidence_threshold": conf_thresh,
        "heif_enabled": HEIF_ENABLED,
    }


""" OPTIONAL LOCAL TEST """
if __name__ == "__main__":
    device = get_device()
    model, device = load_trained_model(device)
    print(f"Model loaded successfully on device: {device}")
    print("web_inference.py is ready for use in Flask.")