
import cv2
import json
import math
import time
import csv

from pathlib import Path


def process_video(
    engine,
    input_path,
    output_path,
    mode="combined",
    frame_stride=3,
    max_frames=None,
    progress_callback=None
):

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if output_path.exists():
        raise FileExistsError(output_path)

    if frame_stride < 1:
        raise ValueError("Invalid frame stride")

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        cap.release()
        raise ValueError("Unable to open video")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if not math.isfinite(source_fps) or source_fps <= 0:
        cap.release()
        raise ValueError("Invalid source video FPS")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_fps = source_fps / frame_stride

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Unable to create video output")

    frame_index = 0
    processed = 0
    records = []

    start = time.perf_counter()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            if frame_index % frame_stride == 0:

                result = engine.process_image(
                    frame,
                    mode=mode,
                    use_context=True
                )

                writer.write(
                    result["annotated_image"]
                )

                records.append({
                    "frame": frame_index,
                    "timestamp_seconds":
                        frame_index / source_fps,
                    **result["summary"]
                })

                processed += 1

                if progress_callback is not None:

                    progress_callback(
                        frame_index + 1,
                        total_frames
                    )

                if (
                    max_frames is not None
                    and processed >= max_frames
                ):
                    break

            frame_index += 1

    finally:

        cap.release()
        writer.release()

    if not records:
        raise RuntimeError("No frames processed")

    elapsed = time.perf_counter() - start

    report = {
        "source_video": input_path.name,

        "source_fps": source_fps,

        "output_fps": output_fps,

        "frame_stride": frame_stride,

        "processed_frames": processed,

        "processing_seconds": elapsed,

        "processed_frames_per_second":
            processed / max(elapsed, 1e-9),

        "helmet_violation_detections":
            sum(
                r["possible_helmet_violations"]
                for r in records
            ),

        "seatbelt_violation_detections":
            sum(
                r["possible_seatbelt_violations"]
                for r in records
            ),

        "unverified_detections":
            sum(
                r["unverified_detections"]
                for r in records
            ),

        "counting_method":
            "per_processed_frame",

        "unique_violation_count": None
    }

    csv_path = output_path.with_suffix(".csv")

    with open(
        csv_path, "w",
        newline="", encoding="utf-8"
    ) as f:

        writer_csv = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys())
        )

        writer_csv.writeheader()
        writer_csv.writerows(records)

    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=4)
    )

    return report
