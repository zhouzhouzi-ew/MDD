from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import (
    APP_ROOT,
    BASELINES,
    COUNT_LABEL,
    MODEL_BENCHMARKS,
    MODEL_SELECTION_STANDARDS,
    VERSION_ROADMAP,
)
from .data import PretrainingData


RECALL_REQUIREMENT = 0.70


def fmt_pct(value: float) -> str:
    return f"{value:.3%}"


def fmt_auc(value: float) -> str:
    return f"{value:.4f}"


def fmt_signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def best_row(df: pd.DataFrame, label_type: str) -> pd.Series:
    return (
        df[df["label_type"] == label_type]
        .sort_values(["pr_auc_median", "roc_auc_median"], ascending=False)
        .iloc[0]
    )


def method_outcomes(df: pd.DataFrame, metric: str, eps: float = 1e-12) -> tuple[int, int, int, int]:
    wide = df.pivot_table(index=["label_type", "indicator_count"], columns="method", values=metric)
    if not {"linear", "scm"}.issubset(set(wide.columns)):
        return 0, 0, 0, 0
    diff = wide["scm"] - wide["linear"]
    scm_wins = int((diff > eps).sum())
    linear_wins = int((diff < -eps).sum())
    ties = int((diff.abs() <= eps).sum())
    return scm_wins, linear_wins, ties, int(len(diff))


def delta_table(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["label_type", "indicator_count"],
        columns="method",
        values=["pr_auc_median", "roc_auc_median"],
        aggfunc="first",
    )
    if not {"linear", "scm"}.issubset(set(wide["pr_auc_median"].columns)):
        return pd.DataFrame()

    out = pd.DataFrame(index=wide.index).reset_index()
    out["x_label"] = out["label_type"] + " · " + out["indicator_count"].map(COUNT_LABEL)
    out["pr_delta"] = (
        wide["pr_auc_median"]["scm"].to_numpy() - wide["pr_auc_median"]["linear"].to_numpy()
    )
    out["roc_delta"] = (
        wide["roc_auc_median"]["scm"].to_numpy() - wide["roc_auc_median"]["linear"].to_numpy()
    )
    return out


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        [
            "label_name",
            "count_label",
            "method_label",
            "best_features",
            "pr_auc_median",
            "pr_lift",
            "roc_auc_median",
            "roc_lift",
            "precision_median",
        ]
    ].copy()
    return out.rename(
        columns={
            "label_name": "定义",
            "count_label": "指标数",
            "method_label": "方法",
            "best_features": "最佳指标组合",
            "pr_auc_median": "PR-AUC",
            "pr_lift": "PR提升",
            "roc_auc_median": "ROC-AUC",
            "roc_lift": "ROC提升",
            "precision_median": "Precision",
        }
    )


def build_feature_cloud_terms(df: pd.DataFrame) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    lift_sum: Counter[str] = Counter()
    for _, row in df.iterrows():
        features = [item.strip() for item in str(row["best_features"]).split("+") if item.strip()]
        for feature in features:
            counter[feature] += 1
            lift_sum[feature] += max(float(row.get("pr_lift", 0.0)), 0.0)

    rows = []
    max_count = max(counter.values(), default=1)
    max_lift = max(lift_sum.values(), default=1.0) or 1.0
    for feature, count in counter.items():
        lift_bonus = lift_sum[feature] / max_lift
        weight = count + lift_bonus
        rows.append(
            {
                "feature": feature,
                "count": count,
                "mean_positive_pr_lift": lift_sum[feature] / max(count, 1),
                "weight": weight,
                "font_size": 16 + 26 * (count / max_count),
            }
        )
    return pd.DataFrame(rows).sort_values(["weight", "feature"], ascending=[False, True]).reset_index(drop=True)


def _qualified_roc(fpr: float, recall: float) -> float:
    d_value = norm.ppf(1 - fpr) - norm.ppf(1 - recall)
    return float(norm.cdf(d_value / np.sqrt(2)))


def _qualified_pr(fpr: float, recall: float, prevalence: float, n: int = 120001) -> float:
    d_value = norm.ppf(1 - fpr) - norm.ppf(1 - recall)
    thresholds = np.linspace(-8.0, d_value + 8.0, n)
    tpr = 1.0 - norm.cdf(thresholds - d_value)
    fpr_values = 1.0 - norm.cdf(thresholds)
    denominator = prevalence * tpr + (1.0 - prevalence) * fpr_values
    precision = np.divide(prevalence * tpr, denominator, out=np.ones_like(denominator), where=denominator > 0)
    return float(np.trapezoid(precision[::-1], tpr[::-1]))


def _precision_at(fpr: float, recall: float, prevalence: float) -> float:
    return prevalence * recall / (prevalence * recall + (1.0 - prevalence) * fpr)


def _recall_at(fpr: float, roc_auc: float) -> float:
    d_value = np.sqrt(2) * norm.ppf(roc_auc)
    return float(1.0 - norm.cdf(norm.ppf(1 - fpr) - d_value))


def clinical_threshold_table(summary: pd.DataFrame) -> pd.DataFrame:
    best_a = best_row(summary, "A")
    best_b = best_row(summary, "B")
    prevalence_a = BASELINES["A"].pr_auc
    prevalence_b = BASELINES["B"].pr_auc
    rows = []
    for fpr in (0.005, 0.01, 0.05):
        rows.append(
            {
                "可承受FPR": f"{fpr:.1%}",
                "达标ROC-AUC": f"{_qualified_roc(fpr, RECALL_REQUIREMENT):.3f}",
                "达标PR-AUC A/B": (
                    f"{_qualified_pr(fpr, RECALL_REQUIREMENT, prevalence_a):.3f} / "
                    f"{_qualified_pr(fpr, RECALL_REQUIREMENT, prevalence_b):.3f}"
                ),
                "工作点Precision A/B": (
                    f"{_precision_at(fpr, RECALL_REQUIREMENT, prevalence_a):.1%} / "
                    f"{_precision_at(fpr, RECALL_REQUIREMENT, prevalence_b):.1%}"
                ),
                "每例真阳需随访假阳 A/B": (
                    f"{(1 - prevalence_a) * fpr / (prevalence_a * RECALL_REQUIREMENT):.1f} / "
                    f"{(1 - prevalence_b) * fpr / (prevalence_b * RECALL_REQUIREMENT):.1f}"
                ),
                "当前模型Recall A/B": (
                    f"{_recall_at(fpr, best_a['roc_auc_median']):.0%} / "
                    f"{_recall_at(fpr, best_b['roc_auc_median']):.0%}"
                ),
            }
        )
    return pd.DataFrame(rows)


def overview_cards(summary: pd.DataFrame) -> list[dict[str, str]]:
    best_a = best_row(summary, "A")
    best_b = best_row(summary, "B")
    pr_scm, pr_linear, pr_tie, pr_total = method_outcomes(summary, "pr_auc_median")
    return [
        {
            "label": "已完成辅助证据",
            "value": "生理指标 + EEG预训练",
            "detail": "生理模型已完成；EEG 源空间预训练已有结果图和消融结果；fMRI/EEG 联合结果待真实数据。",
        },
        {
            "label": "生理指标最佳 PR-AUC",
            "value": f"A {fmt_auc(best_a['pr_auc_median'])} / B {fmt_auc(best_b['pr_auc_median'])}",
            "detail": "定义A阳性12例、定义B阳性93例，结果应按探索性辅助证据解读。",
        },
        {
            "label": "SCM vs Linear",
            "value": f"PR {pr_scm}/{pr_total}",
            "detail": f"PR-AUC中 SCM 胜出 {pr_scm} 次、Linear 胜出 {pr_linear} 次、持平 {pr_tie} 次。",
        },
    ]


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        text = str(value).strip().replace("±", "+/-").replace("卤", "+/-")
        try:
            return float(text.split("+/-")[0])
        except ValueError:
            return None


def _metric_from_json(metrics: dict[str, Any], *keys: str) -> Any:
    current: Any = metrics
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def build_pretraining_overview(data: PretrainingData) -> dict[str, Any]:
    ablation = data.multiband_ablation.copy()
    if not ablation.empty and "dev_auc" in ablation.columns:
        ablation = ablation.sort_values(["dev_auc", "dev_acc"], ascending=False)

    final_test = _metric_from_json(data.best_metrics, "final_test_subject_primary") or {}
    best_version = {
        "version": "V2",
        "title": "Multi-band Multi-view GNN",
        "bands": ", ".join(data.best_metrics.get("bands_used", ["theta", "alpha", "beta"])),
        "fusion": _metric_from_json(data.best_metrics, "cfg", "fusion") or "mean",
        "dev_auc": _safe_float(data.best_metrics.get("val_subj_auc_mean")),
        "dev_std": _safe_float(data.best_metrics.get("val_subj_auc_std")),
        "test_auc": _safe_float(final_test.get("auc")),
        "test_acc": _safe_float(final_test.get("acc")),
        "test_balacc": _safe_float(final_test.get("balacc")),
        "test_mcc": _safe_float(final_test.get("mcc")),
        "params": data.best_metrics.get("params"),
    }
    primary_rows = []
    for _, row in ablation.head(8).iterrows():
        primary_rows.append(
            {
                "version": "V2",
                "run": str(row.get("run", "")),
                "bands": str(row.get("bands", "")),
                "ssl": _safe_float(row.get("ssl")),
                "val_auc": _safe_float(row.get("dev_auc")),
                "val_std": _safe_float(row.get("dev_std")),
                "val_acc": _safe_float(row.get("dev_acc")),
                "test_auc": _safe_float(row.get("test_auc")),
                "params": int(row["params"]) if "params" in row and not pd.isna(row["params"]) else None,
            }
        )

    risk = {
        "validation_auc": best_version["dev_auc"],
        "validation_auc_std": best_version["dev_std"],
        "heldout_auc": best_version["test_auc"],
        "heldout_acc": best_version["test_acc"],
        "message": (
            "V2 在冻结测试集上达到当前最佳结果，但样本量仍小，且传感器域对照受批次混杂影响；"
            "现阶段适合作为融合模型的预训练表征，不直接表述为临床可用模型。"
        ),
    }
    return {
        "primary_rows": primary_rows,
        "best_version": best_version,
        "version_roadmap": _version_roadmap(data),
        "benchmarks": _benchmark_rows(data),
        "selection_standards": MODEL_SELECTION_STANDARDS,
        "domain_ablation": data.domain_ablation,
        "modalities": _modality_status(best_version, data),
        "asset_images": data.asset_images,
        "risk": risk,
        "historical_transfer": {
            "disjoint": _transfer_rows(data.transfer_disjoint),
            "shared": _transfer_rows(data.transfer_shared),
        },
    }


def _version_roadmap(data: PretrainingData) -> list[dict[str, Any]]:
    rows = [dict(item) for item in VERSION_ROADMAP]
    baseline_final = _metric_from_json(data.baseline_metrics, "final_test_subject_primary") or {}
    for row in rows:
        if row["version"] == "V0":
            row["test_auc"] = _safe_float(baseline_final.get("auc")) or row["test_auc"]
            row["acc"] = _safe_float(baseline_final.get("acc")) or row["acc"]
            row["balacc"] = _safe_float(baseline_final.get("balacc")) or row["balacc"]
            row["mcc"] = _safe_float(baseline_final.get("mcc")) or row["mcc"]
        elif row["version"] == "V2":
            final = _metric_from_json(data.best_metrics, "final_test_subject_primary") or {}
            row["dev_auc"] = _safe_float(data.best_metrics.get("val_subj_auc_mean")) or row["dev_auc"]
            row["dev_std"] = _safe_float(data.best_metrics.get("val_subj_auc_std")) or row["dev_std"]
            row["test_auc"] = _safe_float(final.get("auc")) or row["test_auc"]
            row["acc"] = _safe_float(final.get("acc")) or row["acc"]
            row["balacc"] = _safe_float(final.get("balacc")) or row["balacc"]
            row["mcc"] = _safe_float(final.get("mcc")) or row["mcc"]
            row["params"] = data.best_metrics.get("params") or row["params"]
    return rows


def _comparison_best(rows: Any, model: str | None = None) -> dict[str, Any]:
    if isinstance(rows, dict):
        final = rows.get("final_test_subject_primary", {})
        return {
            "dev_auc": _safe_float(rows.get("val_subj_auc_mean")),
            "dev_std": _safe_float(rows.get("val_subj_auc_std")),
            "test_auc": _safe_float(final.get("auc")),
            "acc": _safe_float(final.get("acc")),
            "balacc": _safe_float(final.get("balacc")),
            "mcc": _safe_float(final.get("mcc")),
            "params": rows.get("params"),
        }
    if not isinstance(rows, list):
        return {}
    candidates = [item for item in rows if model is None or item.get("model") == model]
    if not candidates:
        return {}
    best = sorted(candidates, key=lambda item: _safe_float(item.get("test_auc")) or -1, reverse=True)[0]
    return {
        "dev_auc": _safe_float(best.get("dev_auc")),
        "dev_std": _safe_float(best.get("dev_std")),
        "test_auc": _safe_float(best.get("test_auc")),
        "acc": _safe_float(best.get("test_acc")),
        "balacc": _safe_float(best.get("test_balacc")),
        "mcc": _safe_float(best.get("test_mcc")),
    }


def _benchmark_rows(data: PretrainingData) -> list[dict[str, Any]]:
    by_name = {
        "Mumtaz 2017 LR (spectral features)": _comparison_best(
            data.model_comparisons.get("mumtaz2017"), "LR"
        ),
        "Mumtaz 2018 LR (connectivity features)": _comparison_best(
            data.model_comparisons.get("mumtaz2018"), "LR"
        ),
        "NeuralBench SimpleConv": _comparison_best(data.model_comparisons.get("neuralbench")),
        "GCTNet": _comparison_best(data.model_comparisons.get("gctnet")),
    }
    rows = []
    for item in MODEL_BENCHMARKS:
        row = dict(item)
        if not item.get("is_ours") and item["model"] in by_name:
            row.update({key: value for key, value in by_name[item["model"]].items() if value is not None})
        rows.append(row)
    return rows


def _modality_status(best_version: dict[str, Any], data: PretrainingData) -> dict[str, Any]:
    return {
        "eeg": {
            "status": "已完成",
            "summary": (
                f"{best_version['version']} 使用 {best_version['bands']} 三频带源空间 PEC 图，"
                f"冻结测试 AUC {best_version['test_auc']:.3f}。"
            )
            if best_version.get("test_auc") is not None
            else "已形成源空间 PEC 图和 GNN encoder 产物。",
            "artifacts": len(data.asset_images),
        },
        "fmri": {
            "status": "待训练",
            "summary": "等待真实配对 fMRI/EEG 数据后训练影像分支，并评估融合增益。",
            "artifacts": 0,
        },
    }


def _transfer_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows = []
    for _, row in df.iterrows():
        model = str(row.get("model", ""))
        if model.startswith("delta"):
            continue
        rows.append(
            {
                "model": model,
                "auc": str(row.get("auc", "")),
                "acc": str(row.get("acc", "")),
                "f1": str(row.get("f1", "")),
                "sens": str(row.get("sens", "")),
                "spec": str(row.get("spec", "")),
            }
        )
    return rows


def dashboard_python_files(app_root: Path = APP_ROOT) -> list[Path]:
    files = [app_root / "app.py"]
    dashboard_dir = app_root / "dashboard"
    if dashboard_dir.exists():
        files.extend(sorted(dashboard_dir.glob("*.py")))
    return files
