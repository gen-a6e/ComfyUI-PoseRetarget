#!/usr/bin/env python3
"""Generate numeric and interactive RTMW3D inspection reports."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rtmw3d_inspector import (  # noqa: E402
    build_payload,
    load_image,
    output_directory,
    run_inference,
    write_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RTMW3D-X on one image and generate an overlay, complete "
            "XYZ/score data, and a standalone interactive 3D viewer."))
    parser.add_argument("image", type=Path, help="input image")
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY_ROOT / "rtmw3d_reports",
        help="report root (default: repository/rtmw3d_reports)")
    parser.add_argument(
        "--device", default="auto",
        help="auto, cpu, cuda, cuda:N, mps, or rocm (default: auto)")
    parser.add_argument(
        "--bbox-mode", choices=("auto", "full-image"), default="auto",
        help=("auto uses YOLOX and falls back to the full image; "
              "full-image skips person detection"))
    parser.add_argument(
        "--person-index", type=int, default=0,
        help="person to export when multiple people are found (default: 0)")
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="visibility threshold from 0 to 1 (default: 0.3)")
    parser.add_argument(
        "--model", help="local RTMW3D ONNX path or URL")
    parser.add_argument(
        "--detector", help="local YOLOX ONNX/ZIP path or URL")
    parser.add_argument(
        "--cache-dir", type=Path,
        help="override rtmlib's model cache root")
    parser.add_argument(
        "--open", action="store_true",
        help="open viewer.html in the default browser")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.person_index < 0:
        parser.error("--person-index must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"error: image does not exist: {image_path}", file=sys.stderr)
        return 2

    try:
        print(f"Loading image: {image_path}")
        image = load_image(image_path)
        print(
            "Running RTMW3D-X. The first run downloads a large pose model "
            "and can take several minutes...")
        keypoints, scores, keypoints_2d, metadata = run_inference(
            image,
            device=args.device,
            bbox_mode=args.bbox_mode,
            model=args.model,
            detector=args.detector,
            cache_dir=args.cache_dir,
        )
        payload = build_payload(
            image_path,
            image.shape[1],
            image.shape[0],
            keypoints,
            scores,
            keypoints_2d,
            person_index=args.person_index,
            confidence_threshold=args.confidence,
            inference_metadata=metadata,
        )
        destination = output_directory(
            args.output_root.expanduser().resolve(), image_path)
        files = write_artifacts(
            destination,
            image,
            payload,
            Path(__file__).with_name("rtmw3d_viewer.html"),
        )
    except (IndexError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Detected people: {payload['person_count']}")
    print(
        "Visible keypoints: "
        f"{payload['summary']['visible_count']}/133 "
        f"(threshold {payload['confidence_threshold']:.2f})")
    print("Reports:")
    for name in ("viewer", "overlay", "json", "csv"):
        print(f"  {name:7s} {files[name]}")
    print("Depth convention: negative Z = nearer, positive Z = farther")

    if args.open:
        webbrowser.open(files["viewer"].resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
