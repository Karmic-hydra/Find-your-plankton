from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CNN and Traditional ML experiment outputs.")
    parser.add_argument("--traditional-metrics", default="artifacts/models/traditional/metrics.json")
    parser.add_argument("--cnn-metrics", default="artifacts/models/cnn/metrics.json")
    parser.add_argument("--out-json", default="reports/comparison/head_to_head.json")
    parser.add_argument("--out-md", default="reports/comparison/head_to_head.md")
    args = parser.parse_args()

    traditional = load_json(Path(args.traditional_metrics))
    cnn = load_json(Path(args.cnn_metrics))

    models = traditional.get("models", [])
    if not models:
        raise ValueError("No traditional models found in metrics.json")

    best_traditional = max(models, key=lambda x: x["test_metrics"].get("macro_f1", float("-inf")))

    summary = {
        "traditional_best": {
            "model": best_traditional["model"],
            "test_metrics": best_traditional["test_metrics"],
        },
        "cnn": {
            "test_metrics": cnn.get("test_metrics", {}),
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    t = summary["traditional_best"]
    c = summary["cnn"]
    lines = [
        "# CNN vs Traditional ML: Head-to-Head",
        "",
        f"- Best traditional model: {t['model']}",
        "",
        "## Test Metrics",
        "| Metric | Traditional (Best) | CNN |",
        "|---|---:|---:|",
        f"| Top-1 Accuracy | {t['test_metrics'].get('top1_accuracy', 'NA')} | {c['test_metrics'].get('top1_accuracy', 'NA')} |",
        f"| Top-5 Accuracy | {t['test_metrics'].get('top5_accuracy', 'NA')} | {c['test_metrics'].get('top5_accuracy', 'NA')} |",
        f"| Macro F1 | {t['test_metrics'].get('macro_f1', 'NA')} | {c['test_metrics'].get('macro_f1', 'NA')} |",
        f"| Weighted F1 | {t['test_metrics'].get('weighted_f1', 'NA')} | {c['test_metrics'].get('weighted_f1', 'NA')} |",
        f"| Balanced Accuracy | {t['test_metrics'].get('balanced_accuracy', 'NA')} | {c['test_metrics'].get('balanced_accuracy', 'NA')} |",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote comparison report: {out_md}")


if __name__ == "__main__":
    main()
