
import cv2
import json
import time
import hashlib
import threading
import numpy as np
import torch

from pathlib import Path
from ultralytics import YOLO
from collections import Counter


class TrafficGuardEngine:

    def __init__(self, project_root, device=None):

        self.root = Path(project_root).resolve()

        config_path = (
            self.root / "configs/app_config.json"
        )

        self.config = json.loads(
            config_path.read_text()
        )

        self.device = (
            device if device is not None
            else (0 if torch.cuda.is_available() else "cpu")
        )

        self.models = {}
        self.lock = threading.RLock()

        expected_names = {
            "helmet": {
                0: "with_helmet",
                1: "without_helmet"
            },
            "seatbelt": {
                0: "with_seatbelt",
                1: "without_seatbelt"
            }
        }

        for task, cfg in self.config["models"].items():

            path = self.root / cfg["weights"]

            if not path.is_file():
                raise FileNotFoundError(path)

            sha = hashlib.sha256()

            with open(path, "rb") as f:
                for chunk in iter(
                    lambda: f.read(1024**2), b""
                ):
                    sha.update(chunk)

            if sha.hexdigest() != cfg["model_sha256"]:
                raise ValueError(
                    f"Model integrity failed: {task}"
                )

            self.models[task] = YOLO(str(path))

            if task in expected_names:

                if self.models[task].names != expected_names[task]:
                    raise ValueError(
                        f"Class mapping mismatch: {task}"
                    )


    @staticmethod
    def area(box):

        return (
            max(0, box[2] - box[0])
            * max(0, box[3] - box[1])
        )


    @staticmethod
    def intersection(a, b):

        return (
            max(
                0,
                min(a[2], b[2]) - max(a[0], b[0])
            )
            *
            max(
                0,
                min(a[3], b[3]) - max(a[1], b[1])
            )
        )


    @classmethod
    def coverage(cls, inner, outer):

        return cls.intersection(inner, outer) / max(
            cls.area(inner), 1e-9
        )


    @staticmethod
    def horizontal_overlap(a, b):

        overlap = max(
            0,
            min(a[2], b[2]) - max(a[0], b[0])
        )

        return overlap / max(
            min(a[2]-a[0], b[2]-b[0]),
            1e-9
        )


    def predict_model(self, task, image):

        cfg = self.config["models"][task]

        kwargs = {
            "source": image,
            "imgsz": cfg["image_size"],
            "conf": cfg["confidence"],
            "iou": cfg["nms_iou"],
            "max_det": cfg.get("max_det", 300),
            "device": self.device,
            "batch": 1,
            "verbose": False,
            "save": False,
            "stream": False
        }

        if task == "context":
            kwargs["classes"] = cfg["classes"]

        with self.lock, torch.inference_mode():

            result = self.models[task].predict(
                **kwargs
            )[0]

            detections = []

            if result.boxes is not None:

                for row in (
                    result.boxes.data
                    .detach().cpu().numpy()
                ):

                    x1, y1, x2, y2, conf, cls = row

                    cls = int(cls)

                    detections.append({
                        "task": task,
                        "class_id": cls,
                        "label": self.models[task].names[cls],
                        "confidence": float(conf),
                        "bbox": [
                            float(x1), float(y1),
                            float(x2), float(y2)
                        ]
                    })

        return detections


    def associate_helmet(self, detection, context):

        head = detection["bbox"]

        people = [
            d for d in context
            if d["class_id"] == 0
        ]

        motorcycles = [
            d for d in context
            if d["class_id"] == 3
        ]

        settings = self.config["context_association"]

        matches = []

        for person in people:

            body = person["bbox"]

            body_height = body[3] - body[1]

            head_y = (head[1] + head[3]) / 2

            # The helmet/head must be located
            # in the upper portion of a person.

            if head_y > body[1] + 0.45 * body_height:
                continue

            if self.coverage(head, body) < (
                settings["person_head_coverage"]
            ):
                continue

            for motorcycle in motorcycles:

                bike = motorcycle["bbox"]

                if self.horizontal_overlap(body, bike) < (
                    settings[
                        "person_vehicle_horizontal_overlap"
                    ]
                ):
                    continue

                # Require the person's lower body
                # to overlap the motorcycle region.

                lower_body = [
                    body[0],
                    body[1] + 0.45 * body_height,
                    body[2],
                    body[3]
                ]

                if self.intersection(
                    lower_body, bike
                ) <= 0:
                    continue

                matches.append(motorcycle)

        if matches:
            return "motorcycle_context_supported"

        return "context_unverified"


    def associate_seatbelt(self, detection, context):

        occupant = detection["bbox"]

        vehicles = [
            d for d in context
            if d["class_id"] in [2, 5, 7]
        ]

        minimum = self.config[
            "context_association"
        ]["vehicle_occupant_coverage"]

        for vehicle in vehicles:

            if self.coverage(
                occupant,
                vehicle["bbox"]
            ) >= minimum:

                return "vehicle_context_supported"

        return "context_unverified"


    def annotate(self, image, detections):

        output = image.copy()

        height, width = output.shape[:2]

        for d in detections:

            x1, y1, x2, y2 = map(
                int, d["bbox"]
            )

            x1 = max(0, min(width-1, x1))
            x2 = max(0, min(width-1, x2))
            y1 = max(0, min(height-1, y1))
            y2 = max(0, min(height-1, y2))

            if d["status"] == "context_unverified":

                color = (0, 165, 255)

            elif d["class_id"] == 1:

                color = (0, 0, 230)

            else:

                color = (0, 180, 0)

            label = (
                f'{d["label"]} '
                f'{d["confidence"]:.2f}'
            )

            if d["status"] == "context_unverified":
                label += " | review"

            elif d["class_id"] == 1:
                label += " | possible"

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                output,
                label,
                (x1, max(15, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

        return output


    def process_image(
        self,
        image,
        mode="combined",
        use_context=True
    ):

        if (
            image is None
            or not isinstance(image, np.ndarray)
            or image.size == 0
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError(
                "Expected a valid three-channel BGR image."
            )

        if mode not in {
            "helmet", "seatbelt", "combined"
        }:
            raise ValueError("Unsupported detection mode")

        start = time.perf_counter()

        context = (
            self.predict_model("context", image)
            if use_context else []
        )

        tasks = (
            ["helmet", "seatbelt"]
            if mode == "combined"
            else [mode]
        )

        detections = []

        for task in tasks:

            rows = self.predict_model(task, image)

            for row in rows:

                if not use_context:

                    row["status"] = "context_unverified"

                elif task == "helmet":

                    row["status"] = (
                        self.associate_helmet(
                            row, context
                        )
                    )

                else:

                    row["status"] = (
                        self.associate_seatbelt(
                            row, context
                        )
                    )

                row["possible_violation"] = (
                    row["class_id"] == 1
                    and row["status"]
                    != "context_unverified"
                )

                detections.append(row)

        counts = Counter(
            d["label"] for d in detections
        )

        summary = {
            "mode": mode,
            "total_detections": len(detections),

            "with_helmet":
                counts["with_helmet"],

            "without_helmet":
                counts["without_helmet"],

            "with_seatbelt":
                counts["with_seatbelt"],

            "without_seatbelt":
                counts["without_seatbelt"],

            "possible_helmet_violations":
                sum(
                    d["possible_violation"]
                    and d["task"] == "helmet"
                    for d in detections
                ),

            "possible_seatbelt_violations":
                sum(
                    d["possible_violation"]
                    and d["task"] == "seatbelt"
                    for d in detections
                ),

            "unverified_detections":
                sum(
                    d["status"] == "context_unverified"
                    for d in detections
                )
        }

        annotated = self.annotate(
            image, detections
        )

        summary["processing_ms"] = (
            time.perf_counter() - start
        ) * 1000

        return {
            "annotated_image": annotated,
            "detections": detections,
            "context_objects": context,
            "summary": summary
        }
