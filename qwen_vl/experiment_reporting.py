import json
import os
from typing import Any, Dict, List


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return correct / total


def build_agreement_rows(
    rows: List[Dict[str, Any]],
    full_correct_key: str = "full_image_correct",
    embed_correct_key: str = "embedding_only_correct",
) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        full_ok = bool(row.get(full_correct_key, False))
        embed_ok = bool(row.get(embed_correct_key, False))
        output.append(
            {
                "sample_key": row.get("sample_key"),
                "full_image_correct": full_ok,
                "embedding_only_correct": embed_ok,
                "agreement": full_ok == embed_ok,
                "case": (
                    "both_correct"
                    if full_ok and embed_ok
                    else "full_only"
                    if full_ok and not embed_ok
                    else "embed_only"
                    if embed_ok and not full_ok
                    else "both_wrong"
                ),
                "full_image_prediction": row.get("full_image_prediction"),
                "embedding_only_prediction": row.get("embedding_only_prediction"),
                "ground_truth": row.get("ground_truth"),
            }
        )
    return output
