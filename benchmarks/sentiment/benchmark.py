"""Run the same synthetic ticket set through rules, HF models, and int8 ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rules import LABELS, classify


CASE_FILE = Path(__file__).with_name("cases.json")
MODEL_CACHE = Path(tempfile.gettempdir()) / "mlai26-sentiment-hf-cache"
ONNX_CACHE = Path(tempfile.gettempdir()) / "mlai26-sentiment-onnx-final"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))

MODELS = (
    {
        "name": "phobert",
        "model_id": "wonrax/phobert-base-vietnamese-sentiment",
        "revision": "9076a5896971b5d551588fe8a51c722c89731d36",
        "segment_vietnamese": True,
        "license": "MIT",
        "training_data": "30K e-commerce reviews (model card)",
    },
    {
        "name": "visobert",
        "model_id": "5CD-AI/Vietnamese-Sentiment-visobert",
        "revision": "d7738b85200272f10ba8b048d80236c8c582b581",
        "segment_vietnamese": False,
        "license": "Not declared in the pinned model card",
        "training_data": "Vietnamese sentiment datasets (see model card)",
    },
)


def load_cases() -> list[dict[str, str]]:
    cases = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Sentiment case IDs must be unique.")
    counts = Counter(case["label"] for case in cases)
    if set(counts) != set(LABELS) or len(set(counts.values())) != 1:
        raise ValueError("Cases must have a balanced NEGATIVE/NEUTRAL/POSITIVE split.")
    for case in cases:
        if not case["text"].strip():
            raise ValueError(f"Case {case['id']} has empty text.")
    return cases


def score(cases: list[dict[str, str]], predictions: list[str]) -> dict[str, Any]:
    expected = [case["label"] for case in cases]
    if len(expected) != len(predictions):
        raise ValueError("Prediction count does not match case count.")

    confusion = {label: {column: 0 for column in LABELS} for label in LABELS}
    for actual, predicted in zip(expected, predictions, strict=True):
        if predicted not in LABELS:
            raise ValueError(f"Unsupported sentiment label: {predicted}")
        confusion[actual][predicted] += 1

    per_label: dict[str, dict[str, float]] = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in LABELS if other != label)
        false_negative = sum(confusion[label][other] for other in LABELS if other != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}

    errors = [
        {"id": case["id"], "expected": case["label"], "predicted": prediction}
        for case, prediction in zip(cases, predictions, strict=True)
        if case["label"] != prediction
    ]
    total = len(cases)
    return {
        "accuracy": sum(actual == predicted for actual, predicted in zip(expected, predictions, strict=True)) / total,
        "macro_f1": sum(per_label[label]["f1"] for label in LABELS) / len(LABELS),
        "negative_precision": per_label["NEGATIVE"]["precision"],
        "negative_recall": per_label["NEGATIVE"]["recall"],
        "per_label": per_label,
        "confusion_matrix": confusion,
        "errors": errors,
        "error_count": len(errors),
    }


def percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    p50_index = min(len(ordered) - 1, round(0.50 * (len(ordered) - 1)))
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "p50_ms": round(ordered[p50_index], 2),
        "p95_ms": round(ordered[p95_index], 2),
    }


def rules_result(cases: list[dict[str, str]]) -> dict[str, Any]:
    predictions: list[str] = []
    explanations: dict[str, dict[str, list[str]]] = {}
    latencies: list[float] = []
    for case in cases:
        start = time.perf_counter()
        label, hits = classify(case["text"])
        latencies.append((time.perf_counter() - start) * 1000)
        predictions.append(label)
        explanations[case["id"]] = hits
    return {
        "metrics": score(cases, predictions),
        "latency": percentiles(latencies),
        "predictions": dict(zip((case["id"] for case in cases), predictions, strict=True)),
        "explanations": explanations,
    }


def normalize_model_label(raw: str) -> str:
    label = raw.lower().replace("label_", "")
    if "neg" in label:
        return "NEGATIVE"
    if "pos" in label:
        return "POSITIVE"
    if "neu" in label:
        return "NEUTRAL"
    raise ValueError(f"Cannot map model label {raw!r}; inspect the pinned model config.")


def prepare_text(text: str, needs_word_segmentation: bool) -> str:
    if not needs_word_segmentation:
        return text
    from underthesea import word_tokenize

    return word_tokenize(text, format="text")


def hf_prediction(
    model_info: dict[str, Any],
    model: Any,
    tokenizer: Any,
    text: str,
    torch: Any,
) -> tuple[str, float]:
    start = time.perf_counter()
    prepared_text = prepare_text(text, model_info["segment_vietnamese"])
    inputs = tokenizer(
        prepared_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
    inference_ms = (time.perf_counter() - start) * 1000
    index = int(logits.argmax().item())
    raw_label = model.config.id2label[index]
    return normalize_model_label(str(raw_label)), inference_ms


def model_size_bytes(folder: Path, suffixes: tuple[str, ...]) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file() and path.suffix in suffixes)


def load_hf_pair(model_info: dict[str, Any]) -> tuple[Any, Any, Path, float]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODEL_CACHE)
    model_path = MODEL_CACHE / f"{model_info['name']}-{model_info['revision']}"
    snapshot_download(
        repo_id=model_info["model_id"],
        revision=model_info["revision"],
        local_dir=str(model_path),
    )
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=not model_info["segment_vietnamese"],
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.to("cpu")
    model.eval()
    return model, tokenizer, model_path, time.perf_counter() - start


def export_quantized_onnx(
    model_info: dict[str, Any],
) -> tuple[Path, int, float | None, bool]:
    from optimum.exporters.onnx import main_export
    from onnxruntime.quantization import QuantType, quantize_dynamic

    export_folder = ONNX_CACHE / model_info["name"]
    quantized_path = export_folder / "model.int8.onnx"
    if quantized_path.exists():
        return quantized_path, quantized_path.stat().st_size, None, True

    export_folder.mkdir(parents=True, exist_ok=True)
    float_folder = export_folder / "float"
    start = time.perf_counter()
    main_export(
        model_info["local_path"],
        output=str(float_folder),
        task="text-classification",
        device="cpu",
    )
    candidates = sorted(float_folder.glob("*.onnx"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one exported graph, found {candidates}")
    quantize_dynamic(
        model_input=str(candidates[0]),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    return quantized_path, quantized_path.stat().st_size, time.perf_counter() - start, False


def evaluate_onnx(
    model_info: dict[str, Any],
    cases: list[dict[str, str]],
    tokenizer: Any,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    model_path, size_bytes, export_seconds, export_cached = export_quantized_onnx(model_info)
    session_start = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    session_load_seconds = time.perf_counter() - session_start
    input_names = {item.name for item in session.get_inputs()}
    predictions: list[str] = []
    latencies: list[float] = []
    first_inference_ms: float | None = None
    for case in cases:
        start = time.perf_counter()
        prepared = prepare_text(case["text"], model_info["segment_vietnamese"])
        encoded = tokenizer(
            prepared,
            return_tensors="np",
            truncation=True,
            max_length=256,
        )
        inputs = {
            key: value.astype(np.int64)
            for key, value in encoded.items()
            if key in input_names
        }
        outputs = session.run(None, inputs)[0][0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        if first_inference_ms is None:
            first_inference_ms = elapsed_ms
        index = int(np.argmax(outputs))
        config = model_info.get("id2label", {})
        if not config:
            raise ValueError(f"Missing label mapping for {model_info['name']}.")
        predictions.append(normalize_model_label(str(config.get(index, config.get(str(index))))))
    return {
        "metrics": score(cases, predictions),
        "latency": {
            "export_quantize_seconds": round(export_seconds, 2) if export_seconds is not None else None,
            "session_load_seconds": round(session_load_seconds, 3),
            "first_inference_ms": round(first_inference_ms or 0.0, 2),
            "export_cached": export_cached,
            **percentiles(latencies),
        },
        "quantized_model_bytes": size_bytes,
        "predictions": dict(zip((case["id"] for case in cases), predictions, strict=True)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases()
    results: dict[str, Any] = {"rules": rules_result(cases)}
    if not args.rules_only:
        for model_info in MODELS:
            model, tokenizer, model_path, load_seconds = load_hf_pair(model_info)
            model_info["local_path"] = str(model_path)
            import torch

            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            predictions: list[str] = []
            latencies: list[float] = []
            for case in cases:
                prediction, elapsed_ms = hf_prediction(model_info, model, tokenizer, case["text"], torch)
                predictions.append(prediction)
                latencies.append(elapsed_ms)
            model_info["id2label"] = {
                str(key): str(value) for key, value in model.config.id2label.items()
            }
            results[model_info["name"]] = {
                "metrics": score(cases, predictions),
                "latency": {
                    "load_seconds": round(load_seconds, 2),
                    "first_inference_ms": round(latencies[0], 2) if latencies else 0.0,
                    **percentiles(latencies),
                },
                "checkpoint_bytes": model_size_bytes(model_path, (".bin", ".safetensors")),
                "predictions": dict(zip((case["id"] for case in cases), predictions, strict=True)),
                "label_mapping": model_info["id2label"],
            }
            if not args.skip_onnx:
                results[f"{model_info['name']}_onnx_int8"] = evaluate_onnx(model_info, cases, tokenizer)

        for model_info in MODELS:
            full_precision = results[model_info["name"]]["metrics"]
            quantized = results[f"{model_info['name']}_onnx_int8"]["metrics"]
            quantized["accuracy_delta_vs_full_precision"] = round(
                quantized["accuracy"] - full_precision["accuracy"], 4
            )
            quantized["macro_f1_delta_vs_full_precision"] = round(
                quantized["macro_f1"] - full_precision["macro_f1"], 4
            )

    case_bytes = CASE_FILE.read_bytes()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "file": "benchmarks/sentiment/cases.json",
            "sha256": hashlib.sha256(case_bytes).hexdigest(),
            "count": len(cases),
            "label_counts": dict(Counter(case["label"] for case in cases)),
            "synthetic_only": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "CPU",
            "cpu_count": os.cpu_count(),
            "torch_version": None if args.rules_only else __import__("torch").__version__,
            "transformers_version": None if args.rules_only else __import__("transformers").__version__,
            "onnxruntime_version": None if args.rules_only else __import__("onnxruntime").__version__,
            "underthesea_version": None if args.rules_only else __import__("importlib.metadata", fromlist=["version"]).version("underthesea"),
        },
        "models": {
            model_info["name"]: {
                "id": model_info["model_id"],
                "revision": model_info["revision"],
                "license": model_info["license"],
                "training_data": model_info["training_data"],
                "requires_word_segmentation": model_info["segment_vietnamese"],
            }
            for model_info in MODELS
        },
        "limitations": [
            "All cases were authored for this benchmark and are not a representative or independent production test set.",
            "Sentiment describes tone; it does not determine urgency, risk, priority, or policy action.",
            "Model scores are not calibrated confidence values.",
        ],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-only", action="store_true", help="Skip downloads and transformer inference.")
    parser.add_argument("--skip-onnx", action="store_true", help="Evaluate PyTorch checkpoints but skip ONNX export/inference.")
    parser.add_argument("--output", default=str(Path(__file__).with_name("results.latest.json")))
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, result in report["results"].items():
        metrics = result["metrics"]
        latency = result["latency"]
        print(
            f"{name:20} accuracy={metrics['accuracy']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} "
            f"negative_precision={metrics['negative_precision']:.3f} "
            f"negative_recall={metrics['negative_recall']:.3f} "
            f"p50={latency['p50_ms']:.2f}ms p95={latency['p95_ms']:.2f}ms"
        )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
