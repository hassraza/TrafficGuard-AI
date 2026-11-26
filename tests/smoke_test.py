import sys
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))


def sha256_file(path):

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(lambda: f.read(1024**2), b""):
            sha.update(chunk)

    return sha.hexdigest()


def main():

    config_path = ROOT / "configs/app_config.json"

    assert config_path.is_file()

    config = json.loads(config_path.read_text())

    for task, cfg in config["models"].items():

        path = ROOT / cfg["weights"]

        assert path.is_file(), path

        assert sha256_file(path) == cfg["model_sha256"]

        print(f"[PASS] {task} weights verified")

    import numpy as np

    from trafficguard_refined import RefinedTrafficGuard

    engine = RefinedTrafficGuard(
        project_root=ROOT,
        device="cpu",
        profile="standard"
    )

    image = np.zeros(
        (320, 320, 3),
        dtype=np.uint8
    )

    for mode in ["helmet", "seatbelt", "combined"]:

        result = engine.process_image(
            image,
            mode=mode
        )

        assert "summary" in result

        assert "detections" in result

        assert result["annotated_image"].shape == image.shape

        json.dumps(result["summary"])

        print(f"[PASS] {mode} inference")

    try:

        engine.process_image(None)

    except ValueError:

        print("[PASS] Invalid input handling")

    else:

        raise AssertionError("Invalid input accepted")

    print("\nALL REPOSITORY SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
