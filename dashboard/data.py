from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BASELINES, CLEAN_SOURCE_IMAGE_CATALOG, COUNT_LABEL, METHOD_STYLE


@dataclass(frozen=True)
class PretrainingData:
    root: Path
    multiband_ablation: pd.DataFrame
    best_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    model_comparisons: dict[str, Any]
    domain_ablation: list[dict[str, Any]]
    transfer_disjoint: pd.DataFrame
    transfer_shared: pd.DataFrame
    asset_images: list[dict[str, Any]]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text())


def load_physiology_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "scm_kernel_vs_linear_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到结果文件: {path}")

    df = _read_csv(path).copy()
    df["indicator_count"] = df["indicator_count"].astype(int)
    df["method_label"] = df["method"].map(lambda m: METHOD_STYLE.get(m, {"cn": m})["cn"])
    df["label_name"] = df["label_type"].map(
        lambda v: f"{BASELINES[v].name} · {BASELINES[v].description}"
    )
    df["count_label"] = df["indicator_count"].map(COUNT_LABEL)
    df["x_label"] = df["label_type"] + " · " + df["count_label"]
    df["baseline_pr_auc"] = df["label_type"].map(lambda v: BASELINES[v].pr_auc)
    df["baseline_roc_auc"] = df["label_type"].map(lambda v: BASELINES[v].roc_auc)
    df["pr_lift"] = df["pr_auc_median"].astype(float) - df["baseline_pr_auc"]
    df["roc_lift"] = df["roc_auc_median"].astype(float) - df["baseline_roc_auc"]
    return df


def load_pca_coords(results_dir: Path, kind: str, label_type: str) -> pd.DataFrame:
    prefix = "scm_kernel_pca_best_coords" if kind == "best" else "scm_kernel_pca_coords"
    return _read_csv(results_dir / f"{prefix}_type{label_type}.csv")


def load_pca_features(results_dir: Path, kind: str, label_type: str) -> list[str]:
    prefix = "best" if kind == "best" else "low_missing"
    path = results_dir / f"scm_kernel_pca_{prefix}_type{label_type}_loadings.csv"
    if not path.exists() and kind == "low":
        return ["年龄", "BMI", "术前-肌酐"]
    if not path.exists():
        return []
    df = _read_csv(path)
    return df.get("feature", pd.Series(dtype=str)).dropna().astype(str).tolist()


def load_pretraining_data(pretrain_root: Path) -> PretrainingData:
    experiments_dir = pretrain_root / "experiments"
    static_eeg_dir = Path(__file__).resolve().parent / "static" / "eeg"
    asset_dirs = {
        "v2_connectivity_7net.png": static_eeg_dir,
        "v2_node_strength_top24.png": static_eeg_dir,
        "roc_test.png": static_eeg_dir,
        "confusion_test.png": static_eeg_dir,
        "training_curves.png": static_eeg_dir,
    }
    multiband_ablation = _read_csv(experiments_dir / "02_multiband_gnn" / "ablation_table.csv")
    best_metrics = _read_json(
        experiments_dir / "02_multiband_gnn" / "runs" / "_best" / "metrics_summary.json"
    )
    baseline_metrics = _read_json(experiments_dir / "00_baseline_repro" / "metrics_summary.json")
    comparison_dir = experiments_dir / "03_model_comparison" / "runs"
    model_comparisons = {
        "mumtaz2017": _read_json(comparison_dir / "mumtaz2017" / "summary.json"),
        "mumtaz2018": _read_json(comparison_dir / "mumtaz2018" / "summary.json"),
        "neuralbench": _read_json(comparison_dir / "neuralbench" / "summary.json"),
        "gctnet": _read_json(comparison_dir / "gctnet" / "summary.json"),
    }
    domain_ablation = _read_json(comparison_dir / "domain_ablation" / "summary.json")
    transfer_disjoint = _read_csv(pretrain_root / "downstream_results_disjoint.csv")
    transfer_shared = _read_csv(pretrain_root / "downstream_results_shared.csv")
    legacy_image_dir = pretrain_root / "results_plots_source"
    asset_images: list[dict[str, Any]] = []
    for filename, title, caption in CLEAN_SOURCE_IMAGE_CATALOG:
        path = asset_dirs.get(filename, legacy_image_dir) / filename
        if path.exists():
            asset_images.append({"path": path, "title": title, "caption": caption})

    return PretrainingData(
        root=pretrain_root,
        multiband_ablation=multiband_ablation,
        best_metrics=best_metrics,
        baseline_metrics=baseline_metrics,
        model_comparisons=model_comparisons,
        domain_ablation=domain_ablation if isinstance(domain_ablation, list) else [],
        transfer_disjoint=transfer_disjoint,
        transfer_shared=transfer_shared,
        asset_images=asset_images,
    )
