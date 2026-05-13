# coding: utf-8
import os
import os.path as osp
import importlib
import argparse
import time
import uuid

import cv2

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

FR_ENGINES = {
    "liveportrait": {
        "module": "fluxrt.stream_processor.postprocessors.liveportrait",
        "cls": "LivePortraitPostProcessor",
        "kwargs": {
            "models_dir": osp.join(ROOT, "LivePortrait", "liveportrait"),
        },
    },
}


def build_processor(engine: str):
    cfg = FR_ENGINES[engine]
    cls = getattr(importlib.import_module(cfg["module"]), cfg["cls"])
    return cls(**cfg["kwargs"])


def run(source: str, driving: str, output_dir: str, engine: str) -> None:
    processor = build_processor(engine)

    source_rgb = cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB)
    driving_rgb = cv2.cvtColor(cv2.imread(driving), cv2.COLOR_BGR2RGB)

    print("Warming up...")
    processor.process(source_rgb, driving_rgb)

    t0 = time.perf_counter()
    result = processor.process(source_rgb, driving_rgb)
    print(f"Inference: {time.perf_counter() - t0:.3f}s")

    os.makedirs(output_dir, exist_ok=True)
    out_path = osp.join(output_dir, uuid.uuid4().hex + ".jpg")
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Done. Result: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Face to modify")
    parser.add_argument("driving", help="Reference expression")
    parser.add_argument("--engine", choices=list(FR_ENGINES), default="liveportrait")
    parser.add_argument("--output-dir", default="output/face_reenactment")
    opt = parser.parse_args()
    run(opt.source, opt.driving, opt.output_dir, opt.engine)
