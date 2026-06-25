from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Sequence
from retinaface.pre_trained_models import get_model
import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


SUPPORTED_DATASETS = ("FF", "DFD", "DFDC", "CDF", "FFIW", "DFDCP")


@dataclass(frozen=True)
class DatasetPaths:
    faceforensics_root: str
    dfd_real_dir: str
    dfd_fake_dir: str
    dfdc_root: str
    cdf_root: str
    dfdcp_root: str
    ffiw_root: str

def _resolve_dataset_names(datasets: str | Sequence[str]) -> list[str]:
    if isinstance(datasets, str):
        if datasets.lower() == "all":
            return list(SUPPORTED_DATASETS)
        datasets = [datasets]

    resolved = []
    for dataset_name in datasets:
        if dataset_name not in SUPPORTED_DATASETS:
            valid = ", ".join(SUPPORTED_DATASETS)
            raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {valid}")
        resolved.append(dataset_name)

    return resolved

def get_faceforensics_test_split(
    paths: DatasetPaths,
    manipulation: str = "all",
    phase: str = "test",
) -> tuple[list[str], list[int]]:
    valid_manipulations = {"all", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}
    if manipulation not in valid_manipulations:
        raise ValueError(f"Unknown FaceForensics manipulation '{manipulation}'.")

    root = Path(paths.faceforensics_root)
    original_dir = root / "original_sequences" / "youtube" / "raw" / "videos"
    split_file = root / f"{phase}.json"

    with split_file.open("r") as f:
        split_pairs = json.load(f)

    video_ids = {video_id for pair in split_pairs for video_id in pair}

    real_videos = sorted(str(p) for p in original_dir.glob("*.mp4") if p.stem[:3] in video_ids)
    real_labels = [0] * len(real_videos)

    if manipulation == "all":
        manipulations = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    else:
        manipulations = [manipulation]

    fake_videos: list[str] = []
    for method in manipulations:
        fake_dir = root / "manipulated_sequences" / method / "raw" / "videos"
        fake_videos.extend(
            str(p) for p in sorted(fake_dir.glob("*.mp4")) if p.stem[:3] in video_ids
        )

    fake_labels = [1] * len(fake_videos)
    return real_videos + fake_videos, real_labels + fake_labels


def get_dfd_test_split(paths: DatasetPaths) -> tuple[list[str], list[int]]:
    real_videos = sorted(glob(str(Path(paths.dfd_real_dir) / "*.mp4")))
    fake_videos = sorted(glob(str(Path(paths.dfd_fake_dir) / "*.mp4")))

    videos = real_videos + fake_videos
    labels = [0] * len(real_videos) + [1] * len(fake_videos)
    return videos, labels

def get_dfdc_test_split(paths: DatasetPaths) -> tuple[list[str], list[int]]:
    root = Path(paths.dfdc_root)
    labels_file = root / "test" / "labels.csv"
    videos_dir = root / "test" / "videos"

    videos: list[str] = []
    labels: list[int] = []

    with labels_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append(str(videos_dir / row["filename"]))
            labels.append(int(row["label"]))

    return videos, labels

def get_cdf_test_split(paths: DatasetPaths) -> tuple[list[str], list[int]]:
    root = Path(paths.cdf_root)
    split_file = root / "List_of_testing_videos.txt"

    videos: list[str] = []
    labels: list[int] = []

    with split_file.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            label_str, relative_path = parts
            subset_name, video_name = relative_path.split("/")
            videos.append(str(root / subset_name / "videos" / video_name))
            labels.append(1 - int(label_str))

    return videos, labels

def get_dfdcp_test_split(paths: DatasetPaths, phase: str = "test") -> tuple[list[str], list[int]]:
    root = Path(paths.dfdcp_root)
    phase_map = {"train": "train", "val": "train", "test": "test"}

    metadata_file = root / "dataset.json"
    with metadata_file.open("r") as f:
        metadata = json.load(f)

    selected = {
        Path(key).name: int(item["label"] == "fake")
        for key, item in metadata.items()
        if item["set"] == phase_map[phase]
    }

    candidate_files = glob(str(root / "method_*/*/*/*.mp4"))
    candidate_files += glob(str(root / "original_videos/*/*.mp4"))
    candidate_files = sorted(candidate_files)

    videos = [p for p in candidate_files if Path(p).name in selected]
    labels = [selected[Path(p).name] for p in videos]
    return videos, labels

def get_ffiw_test_split(paths: DatasetPaths) -> tuple[list[str], list[int]]:
    root = Path(paths.ffiw_root)
    source_files = sorted(glob(str(root / "source" / "test" / "*.mp4")))
    target_files = sorted(glob(str(root / "target" / "test" / "*.mp4")))

    videos = source_files + target_files
    labels = [0] * len(source_files) + [1] * len(target_files)
    return videos, labels

def load_dataset_split(dataset_name: str, paths: DatasetPaths) -> tuple[list[str], list[int]]:
    loaders = {
        "FF": get_faceforensics_test_split,
        "DFD": get_dfd_test_split,
        "DFDC": get_dfdc_test_split,
        "CDF": get_cdf_test_split,
        "FFIW": get_ffiw_test_split,
        "DFDCP": get_dfdcp_test_split,
    }

    return loaders[dataset_name](paths)

def crop_face_from_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    margin_ratio: float = 0.125,
) -> np.ndarray:
    """
    Crop a face region from an RGB image using a bounding box.
    """
    height, width = image.shape[:2]

    (x0, y0), (x1, y1) = bbox.astype(np.float32)
    box_width = x1 - x0
    box_height = y1 - y0

    x_margin = box_width * margin_ratio
    y_margin = box_height * margin_ratio

    x0 = max(0, int(x0 - x_margin))
    y0 = max(0, int(y0 - y_margin))
    x1 = min(width, int(x1 + x_margin) + 1)
    y1 = min(height, int(y1 + y_margin) + 1)

    return image[y0:y1, x0:x1]

def extract_faces_from_video(
    filename: str,
    num_frames: int,
    face_detector,
    image_size: int,
) -> tuple[list[np.ndarray], list[int]]:

    capture = cv2.VideoCapture(filename)

    if not capture.isOpened():
        print(f"Cannot open: {filename}")
        return [], []

    face_crops: list[np.ndarray] = []
    frame_indices: list[int] = []

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        return [], []

    sampled_indices = np.linspace(
        0,
        total_frames - 1,
        num=num_frames,
        endpoint=True,
        dtype=int,
    )

    sampled_set = set(int(i) for i in sampled_indices)

    for frame_idx in range(total_frames):
        ok, frame_bgr = capture.read()

        if not ok or frame_bgr is None:
            tqdm.write(f"Frame read {frame_idx} Error! : {Path(filename).name}")
            break

        if frame_idx not in sampled_set:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detections = face_detector.predict_jsons(frame_rgb)

        try:
            if len(detections) == 0:
                tqdm.write(f"No faces in {frame_idx}:{Path(filename).name}")
                continue

            current_faces: list[np.ndarray] = []
            current_indices: list[int] = []
            current_areas: list[float] = []

            for detection in detections:
                x0, y0, x1, y1 = detection["bbox"]
                bbox = np.array([[x0, y0], [x1, y1]], dtype=np.float32)

                cropped_face = crop_face_from_bbox(
                    frame_rgb,
                    bbox,
                    margin_ratio=0.125,
                )

                resized_face = cv2.resize(
                    cropped_face,
                    dsize=(image_size, image_size),
                )

                current_faces.append(resized_face.transpose((2, 0, 1)))
                current_indices.append(frame_idx)
                current_areas.append((x1 - x0) * (y1 - y0))

            max_area = max(current_areas)

            for face, idx, area in zip(current_faces, current_indices, current_areas):
                if area >= max_area / 2:
                    face_crops.append(face)
                    frame_indices.append(idx)

        except Exception as exc:
            print(f"error in {frame_idx}:{filename}")
            print(exc)
            continue

    capture.release()
    return face_crops, frame_indices

def aggregate_video_score(face_scores: torch.Tensor, frame_indices: Sequence[int]) -> float:
    """
    Aggregate face scores into a single video score.

    For each frame, take the maximum score across detected faces.
    Then average those frame-level scores across the video.
    """
    if face_scores.numel() == 0:
        return 0.5

    grouped_scores: list[list[float]] = []
    current_frame_idx = None

    for score, frame_idx in zip(face_scores.tolist(), frame_indices):
        if frame_idx != current_frame_idx:
            grouped_scores.append([])
            current_frame_idx = frame_idx
        grouped_scores[-1].append(float(score))

    frame_scores = [max(scores) for scores in grouped_scores]
    return float(np.mean(frame_scores)) if frame_scores else 0.5

def predict_video_score(
    model: torch.nn.Module,
    filename: str,
    face_detector,
    image_size: int,
    num_frames: int,
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> float:
    try:
        face_crops, frame_indices = extract_faces_from_video(
            filename=filename,
            num_frames=num_frames,
            face_detector=face_detector,
            image_size=image_size,
        )
    except Exception as exc:
        raise RuntimeError(f"face extraction failed: {exc}") from exc

    if not face_crops:
        return 0.5

    images = torch.as_tensor(np.asarray(face_crops), dtype=torch.float32, device=device) / 255.0

    all_scores = []
    with torch.no_grad():
        for batch in torch.split(images, batch_size):
            try:
                logits = model(batch)
            except Exception as exc:
                raise RuntimeError(f"model inference failed: {exc}") from exc

            if num_classes == 1:
                scores = logits.view(-1)
            else:
                scores = logits[:, 1]

            all_scores.append(scores)

    face_scores = torch.cat(all_scores, dim=0)
    return aggregate_video_score(face_scores, frame_indices)

def _safe_test_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name.strip())

def save_cross_dataset_results(
    test_name: str,
    results: dict[str, dict[str, float | int]],
    output_dir: str = "./results",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{_safe_test_name(test_name)}.txt"

    mean_auc = float(np.mean([item["auc"] for item in results.values()])) if results else float("nan")

    lines = [
        f"Test name: {test_name}",
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Cross-dataset evaluation results",
        "-" * 40,
    ]

    for dataset_name, metrics in results.items():
        lines.extend(
            [
                f"Dataset:      {dataset_name}",
                f"AUC:          {metrics['auc']:.4f}",
                f"Videos:       {int(metrics['num_videos'])}",
                f"Failures:     {int(metrics['num_failures'])}",
                "",
            ]
        )

    lines.extend(
        [
            "-" * 40,
            f"Mean AUC:     {mean_auc:.4f}",
            "",
        ]
    )

    file_path.write_text("\n".join(lines))
    return file_path

def test_cross_dataset(
    model: torch.nn.Module,
    datasets: str | Sequence[str],
    test_name: str,
    dataset_paths: DatasetPaths,
    image_size: int,
    num_classes: int = 2,
    num_frames: int = 32,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    Evaluate a model on one or more cross-dataset benchmarks.

    Args:
        model: Model to evaluate.
        datasets: Dataset name, list of dataset names, or "all".
        test_name: Name used for the saved summary file.
        dataset_paths: Dataset root paths.
        image_size: Face crop size used before model inference.
        num_classes: 1 for single-logit output, 2 for two-class output.
        num_frames: Number of frames sampled per video.
        batch_size: Batch size for face-level inference.
        device: Torch device. Defaults to CUDA if available, otherwise CPU.

    Returns:
        A dictionary mapping dataset name to ROC AUC.
    """
    dataset_names = _resolve_dataset_names(datasets)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

    summary: dict[str, dict[str, float | int]] = {}
    aucs: dict[str, float] = {}

    for dataset_name in dataset_names:
        video_list, target_list = load_dataset_split(dataset_name, dataset_paths)

        video_scores: list[float] = []
        num_failures = 0

        for filename in tqdm(video_list, desc=f"Testing on {dataset_name}"):
            try:
                score = predict_video_score(
                    model=model,
                    filename=filename,
                    face_detector=face_detector,
                    image_size=image_size,
                    num_frames=num_frames,
                    batch_size=batch_size,
                    num_classes=num_classes,
                    device=device,
                )
            except Exception as exc:
                print(f"[Warning] Failed on video '{filename}': {exc}")
                score = 0.5
                num_failures += 1

            video_scores.append(score)

        auc_value = float(roc_auc_score(target_list, video_scores))
        aucs[dataset_name] = auc_value
        summary[dataset_name] = {
            "auc": auc_value,
            "num_videos": len(video_list),
            "num_failures": num_failures,
        }

    save_cross_dataset_results(test_name=test_name, results=summary)
    return aucs