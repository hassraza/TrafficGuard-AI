
import cv2
import torch
import time
import numpy as np

from trafficguard_engine import TrafficGuardEngine


PROFILES = {

    "standard": {
        "helmet_size": 640,
        "helmet_conf": 0.30,
        "seatbelt_size": 640,
        "seatbelt_conf": 0.35,
        "roi": False
    },

    "high_resolution": {
        "helmet_size": 800,
        "helmet_conf": 0.30,
        "seatbelt_size": 640,
        "seatbelt_conf": 0.35,
        "roi": False
    },

    "maximum_resolution": {
        "helmet_size": 960,
        "helmet_conf": 0.30,
        "seatbelt_size": 640,
        "seatbelt_conf": 0.35,
        "roi": False
    },

    "high_sensitivity": {
        "helmet_size": 800,
        "helmet_conf": 0.20,
        "seatbelt_size": 640,
        "seatbelt_conf": 0.35,
        "roi": False
    },

    "region_assisted": {
        "helmet_size": 640,
        "helmet_conf": 0.30,
        "seatbelt_size": 640,
        "seatbelt_conf": 0.35,
        "roi": True
    }
}


class RefinedTrafficGuard(TrafficGuardEngine):

    def __init__(
        self,
        project_root,
        device=None,
        profile="standard"
    ):

        super().__init__(
            project_root,
            device=device
        )

        self.set_profile(profile)

        self._current_context = []


    def set_profile(self, name):

        if name not in PROFILES:
            raise ValueError(
                f"Unknown profile: {name}"
            )

        with self.lock:

            self.profile_name = name
            self.profile = PROFILES[name].copy()


    @classmethod
    def iou(cls, a, b):

        intersection = cls.intersection(a, b)

        union = (
            cls.area(a)
            + cls.area(b)
            - intersection
        )

        return intersection / max(union, 1e-9)


    def suppress_duplicates(
        self,
        detections,
        threshold=0.5
    ):

        kept = []

        ordered = sorted(
            detections,
            key=lambda d: d["confidence"],
            reverse=True
        )

        for detection in ordered:

            duplicate = any(
                detection["class_id"] == old["class_id"]
                and self.iou(
                    detection["bbox"],
                    old["bbox"]
                ) >= threshold

                for old in kept
            )

            if not duplicate:
                kept.append(detection)

        return kept


    def raw_predict(
        self,
        task,
        image,
        size=None,
        confidence=None
    ):

        cfg = self.config["models"][task]

        kwargs = {
            "source": image,
            "imgsz": (
                size if size is not None
                else cfg["image_size"]
            ),
            "conf": (
                confidence if confidence is not None
                else cfg["confidence"]
            ),
            "iou": cfg["nms_iou"],
            "max_det": cfg.get("max_det", 300),
            "device": self.device,
            "batch": 1,
            "verbose": False,
            "save": False
        }

        if task == "context":

            kwargs["classes"] = cfg["classes"]

        with self.lock, torch.inference_mode():

            result = self.models[task].predict(
                **kwargs
            )[0]

            detections = []

            if result.boxes is not None:

                for values in (
                    result.boxes.data
                    .detach()
                    .cpu()
                    .numpy()
                ):

                    x1, y1, x2, y2, conf, cls = values

                    cls = int(cls)

                    detections.append({
                        "task": task,
                        "class_id": cls,
                        "label": self.models[task].names[cls],
                        "confidence": float(conf),
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2)
                        ],
                        "source": "full_image"
                    })

        return detections


    def motorcycle_crops(
        self,
        image,
        context,
        full_detections
    ):

        height, width = image.shape[:2]

        motorcycles = [
            d for d in context
            if d["class_id"] == 3
        ]

        # Limit expensive additional inference.
        motorcycles = sorted(
            motorcycles,
            key=lambda d: d["confidence"],
            reverse=True
        )[:4]

        extra = []

        for motorcycle in motorcycles:

            x1, y1, x2, y2 = motorcycle["bbox"]

            bw = x2-x1
            bh = y2-y1

            # Expand upwards to include riders.
            cx1 = max(0, int(x1-0.40*bw))
            cy1 = max(0, int(y1-1.50*bh))

            cx2 = min(width, int(x2+0.40*bw))
            cy2 = min(height, int(y2+0.10*bh))

            region = [cx1, cy1, cx2, cy2]

            if cx2-cx1 < 32 or cy2-cy1 < 32:
                continue

            # Avoid extra inference when a head
            # detection is already present nearby.
            supported_head = any(
                self.coverage(
                    d["bbox"], region
                ) >= 0.8
                for d in full_detections
            )

            if supported_head:
                continue

            crop = image[cy1:cy2, cx1:cx2]

            rows = self.raw_predict(
                "helmet",
                crop,
                size=640,
                confidence=self.profile["helmet_conf"]
            )

            for row in rows:

                a, b, c, d = row["bbox"]

                row["bbox"] = [
                    a+cx1,
                    b+cy1,
                    c+cx1,
                    d+cy1
                ]

                row["source"] = "motorcycle_crop"

                extra.append(row)

        return extra


    def predict_model(self, task, image):

        if task == "context":

            self._current_context = (
                self.raw_predict(task, image)
            )

            return self._current_context

        rows = self.raw_predict(
            task,
            image,
            size=self.profile[f"{task}_size"],
            confidence=self.profile[f"{task}_conf"]
        )

        if task == "helmet" and self.profile["roi"]:

            extra = self.motorcycle_crops(
                image,
                self._current_context,
                rows
            )

            rows = self.suppress_duplicates(
                rows + extra
            )

        return rows


    def process_image(
        self,
        image,
        mode="combined",
        use_context=True
    ):

        with self.lock:

            self._current_context = []

            result = super().process_image(
                image,
                mode=mode,
                use_context=use_context
            )

            result["summary"]["profile"] = (
                self.profile_name
            )

            result["summary"]["crop_detections"] = sum(
                d.get("source") == "motorcycle_crop"
                for d in result["detections"]
            )

            self._current_context = []

            return result


    def draw_raw(self, image, detections):

        output = image.copy()

        for d in detections:

            x1, y1, x2, y2 = map(
                int, d["bbox"]
            )

            if d["task"] == "context":

                color = (255, 180, 0)

            else:

                color = (
                    (0, 180, 0)
                    if d["class_id"] == 0
                    else (0, 0, 255)
                )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = (
                f'{d["label"]} '
                f'{d["confidence"]:.2f}'
            )

            cv2.putText(
                output,
                label,
                (x1, max(15, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1
            )

        return output
