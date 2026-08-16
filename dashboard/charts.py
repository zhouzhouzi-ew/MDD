from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import COUNT_LABEL, GRID_COLOR, METHOD_STYLE, TEXT_COLOR
from .data import load_pca_coords, load_pca_features
from .services import delta_table


def _base_layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title or "", x=0.01, font=dict(size=20, color="#111827")),
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color=TEXT_COLOR, size=12),
        margin=dict(l=58, r=28, t=78 if title else 36, b=46),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_color="#172033"),
    )
    return fig


def build_lift_chart(df: pd.DataFrame) -> go.Figure:
    x_order = [f"{label} · {COUNT_LABEL[count]}" for label in ["A", "B"] for count in [1, 2, 3]]
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("PR-AUC 相对阳性率基线", "ROC-AUC 相对 0.50 基线"),
        vertical_spacing=0.16,
    )
    for row, metric, raw_col, title in [
        (1, "pr_lift", "pr_auc_median", "PR-AUC"),
        (2, "roc_lift", "roc_auc_median", "ROC-AUC"),
    ]:
        for method in [m for m in ["linear", "scm"] if m in set(df["method"])]:
            sub = df[df["method"] == method].copy()
            sub["x_label"] = pd.Categorical(sub["x_label"], categories=x_order, ordered=True)
            sub = sub.sort_values("x_label")
            style = METHOD_STYLE.get(method, {"cn": method, "color": "#64748b"})
            fig.add_trace(
                go.Bar(
                    x=sub["x_label"],
                    y=sub[metric],
                    name=style["cn"],
                    marker=dict(color=style["color"], line=dict(color="#ffffff", width=0.8)),
                    customdata=np.stack([sub["best_features"], sub[raw_col]], axis=-1),
                    hovertemplate=(
                        "%{x}<br>方法：" + style["cn"] + "<br>"
                        "指标组合：%{customdata[0]}<br>"
                        f"{title}：%{{customdata[1]:.6f}}<br>"
                        "相对基线：%{y:+.6f}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1.2, row=row, col=1)
        fig.update_yaxes(title_text="提升量", tickfont=dict(size=11, color=TEXT_COLOR), gridcolor=GRID_COLOR, row=row, col=1)
        fig.update_xaxes(tickfont=dict(size=12, color=TEXT_COLOR), row=row, col=1)

    fig.update_layout(barmode="group", bargap=0.28, legend=dict(orientation="h", y=1.07, x=1, xanchor="right", title=None))
    return _base_layout(fig, 660, "各方法最佳组合的相对基线提升")


def build_delta_chart(df: pd.DataFrame) -> go.Figure:
    deltas = delta_table(df)
    if deltas.empty:
        return go.Figure()
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("PR-AUC 差值：SCM - Linear", "ROC-AUC 差值：SCM - Linear"),
        vertical_spacing=0.17,
    )
    for row, metric in [(1, "pr_delta"), (2, "roc_delta")]:
        colors = np.where(deltas[metric] >= 0, "#0f766e", "#b91c1c")
        fig.add_trace(
            go.Bar(
                x=deltas["x_label"],
                y=deltas[metric],
                marker_color=colors,
                hovertemplate="%{x}<br>差值：%{y:+.6f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1.1, row=row, col=1)
        fig.update_xaxes(tickfont=dict(size=12, color=TEXT_COLOR), row=row, col=1)
        fig.update_yaxes(title_text="SCM - Linear", gridcolor=GRID_COLOR, tickfont=dict(size=11, color=TEXT_COLOR), row=row, col=1)
    fig.update_layout(showlegend=False)
    return _base_layout(fig, 560, "SCM 与 Linear 的直接差值")


def build_feature_cloud_chart(terms: pd.DataFrame) -> go.Figure:
    if terms.empty:
        return go.Figure()
    terms = terms.head(18).copy()
    n_terms = len(terms)
    angles = np.linspace(0, 3.8 * np.pi, n_terms)
    radius = np.linspace(0.05, 1.0, n_terms)
    terms["x"] = radius * np.cos(angles)
    terms["y"] = radius * np.sin(angles)
    palette = np.array(["#3B82F6", "#8B5CF6", "#EC4899", "#38BDF8", "#A78BFA", "#FDE68A", "#86EFAC"])
    colors = palette[np.arange(n_terms) % len(palette)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=terms["x"],
            y=terms["y"],
            mode="text",
            text=terms["feature"],
            textfont=dict(size=terms["font_size"], color=colors),
            customdata=np.stack([terms["count"], terms["mean_positive_pr_lift"]], axis=-1),
            hovertemplate="指标：%{text}<br>出现次数：%{customdata[0]}<br>平均正向PR提升：%{customdata[1]:.6f}<extra></extra>",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, 420, "最佳组合指标词云")


def build_method_heatmap(df: pd.DataFrame, metric: str = "pr_auc_median") -> go.Figure:
    heat = df.copy()
    heat["row_label"] = heat["label_type"] + " · " + heat["method_label"]
    pivot = heat.pivot_table(index="row_label", columns="count_label", values=metric, aggfunc="first")
    pivot = pivot.reindex(columns=["单指标", "双指标", "三指标"])
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[[0, "#f1f5f9"], [0.5, "#7dd3fc"], [1, "#0f766e"]],
            colorbar=dict(title=metric.replace("_", " ")),
            hovertemplate="%{y}<br>%{x}<br>值：%{z:.6f}<extra></extra>",
        )
    )
    return _base_layout(fig, 380, "定义 × 方法 × 指标数表现热力图")


def build_radar_chart(df: pd.DataFrame) -> go.Figure:
    metrics = ["pr_auc_median", "roc_auc_median", "precision_median", "sensitivity_median"]
    labels = ["PR-AUC", "ROC-AUC", "Precision", "Sensitivity"]
    max_values = {metric: max(float(df[metric].max()), 1e-12) for metric in metrics}
    fig = go.Figure()
    for label_type, color, fill in [
        ("A", "#3B82F6", "rgba(59,130,246,0.20)"),
        ("B", "#8B5CF6", "rgba(139,92,246,0.18)"),
    ]:
        row = df[df["label_type"] == label_type].sort_values(["pr_auc_median", "roc_auc_median"], ascending=False).iloc[0]
        values = [float(row[metric]) / max_values[metric] for metric in metrics]
        values.append(values[0])
        theta = labels + [labels[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                fill="toself",
                fillcolor=fill,
                name=f"定义{label_type}最佳组合",
                line=dict(color=color, width=2.4),
                hovertemplate="%{theta}<br>归一化值：%{r:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor="#FBFDFF",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E4E9F7"),
            angularaxis=dict(gridcolor="#ECE8FF"),
        )
    )
    return _base_layout(fig, 430, "最佳组合多指标雷达图")


def _pca_title(kind_label: str, label_type: str, features: list[str]) -> str:
    feature_text = " + ".join(features) if features else "未找到指标文件"
    return f"{kind_label} · 定义{label_type}<br><sup>{feature_text}</sup>"


def build_pca_chart(results_dir: Path) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            _pca_title("低缺失指标", "A", load_pca_features(results_dir, "low", "A")),
            _pca_title("低缺失指标", "B", load_pca_features(results_dir, "low", "B")),
            _pca_title("最佳组合指标", "A", load_pca_features(results_dir, "best", "A")),
            _pca_title("最佳组合指标", "B", load_pca_features(results_dir, "best", "B")),
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    configs = [("low", {"A": (0.354, 0.334), "B": (0.354, 0.334)}), ("best", {"A": (0.338, 0.215), "B": (0.338, 0.215)})]
    for row_idx, (kind, ratios) in enumerate(configs):
        for col_idx, label_type in enumerate(["A", "B"]):
            coords = load_pca_coords(results_dir, kind, label_type)
            if coords.empty:
                continue
            non_dep = coords[coords["depressed"] == 0]
            dep = coords[coords["depressed"] == 1]
            if len(non_dep) > 800:
                non_dep = non_dep.sample(800, random_state=20260802)
            row, col = row_idx + 1, col_idx + 1
            pc1, pc2 = ratios[label_type]
            fig.add_trace(
                go.Scatter(
                    x=non_dep["PC1"],
                    y=non_dep["PC2"],
                    mode="markers",
                    marker=dict(color="rgba(37, 99, 235, 0.24)", size=4),
                    name="非抑郁",
                    showlegend=(row_idx == 0 and col_idx == 0),
                    hovertemplate="非抑郁<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=dep["PC1"],
                    y=dep["PC2"],
                    mode="markers",
                    marker=dict(color="#e11d48", size=9, symbol="diamond", line=dict(color="#ffffff", width=0.8)),
                    name="抑郁",
                    showlegend=(row_idx == 0 and col_idx == 0),
                    hovertemplate="抑郁<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text=f"PC1 ({pc1:.1%})", gridcolor=GRID_COLOR, row=row, col=col)
            fig.update_yaxes(title_text=f"PC2 ({pc2:.1%})", gridcolor=GRID_COLOR, row=row, col=col)
    fig.update_layout(legend=dict(orientation="h", y=1.05, x=1, xanchor="right", title=None))
    return _base_layout(fig, 780, "PCA 分布概览")


def build_pretraining_ablation_chart(rows: list[dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    labels = df["run"].str.replace("_", " ", regex=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=df["val_auc"],
            error_y=dict(type="data", array=df["val_std"].fillna(0)),
            marker_color="#7C83FD",
            name="验证AUC",
            customdata=np.stack([df["bands"], df["ssl"], df["params"]], axis=-1),
            hovertemplate="run：%{x}<br>频带：%{customdata[0]}<br>SSL λ：%{customdata[1]}<br>参数：%{customdata[2]}<br>验证AUC：%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8", annotation_text="机会线 0.5")
    fig.update_yaxes(title_text="AUC", range=[0, 1], gridcolor=GRID_COLOR)
    fig.update_xaxes(tickangle=-25)
    return _base_layout(fig, 430, "EEG 多频带预训练消融")


def build_version_roadmap_chart(rows: list[dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    colors = np.where(df["status"].eq("当前最佳"), "#7C83FD", "#93C5FD")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["version"],
            y=df["test_auc"],
            mode="lines+markers+text",
            text=df["status"],
            textposition="top center",
            marker=dict(size=np.where(df["status"].eq("当前最佳"), 18, 12), color=colors, line=dict(color="#ffffff", width=2)),
            line=dict(color="#3B82F6", width=3),
            customdata=np.stack([df["title"], df["input"], df["mcc"]], axis=-1),
            hovertemplate="%{x} %{customdata[0]}<br>输入：%{customdata[1]}<br>TEST AUC：%{y:.3f}<br>MCC：%{customdata[2]:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["version"],
            y=df["dev_auc"],
            marker=dict(color="rgba(167,139,250,0.28)", line=dict(color="rgba(124,131,253,0.40)", width=1)),
            name="DEV AUC",
            hovertemplate="%{x}<br>DEV AUC：%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8", annotation_text="机会线 0.5")
    fig.update_yaxes(title_text="AUC", range=[0.3, 1.0], gridcolor=GRID_COLOR)
    fig.update_xaxes(title_text="")
    fig.update_layout(showlegend=False, bargap=0.46)
    return _base_layout(fig, 460, "版本升级路线图")


def build_model_position_chart(rows: list[dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    metrics = [
        ("test_auc", "TEST AUC"),
        ("acc", "Accuracy"),
        ("balacc", "Balanced Accuracy"),
        ("mcc", "MCC"),
    ]
    fig = go.Figure()
    others = df[~df["is_ours"]].copy()
    ours = df[df["is_ours"]].iloc[0]
    for idx, (metric, label) in enumerate(metrics):
        values = others[metric].astype(float)
        min_v = float(values.min())
        max_v = float(values.max())
        ours_v = float(ours[metric])
        fig.add_trace(
            go.Bar(
                y=[label],
                x=[max_v - min_v],
                base=[min_v],
                orientation="h",
                width=0.38,
                marker=dict(color="rgba(124,131,253,0.30)", line=dict(color="#A78BFA", width=1.4)),
                name="现有模型范围" if idx == 0 else None,
                showlegend=idx == 0,
                hovertemplate=f"{label}<br>现有模型范围：{min_v:.3f} - {max_v:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                y=[label],
                x=[ours_v],
                mode="markers",
                marker=dict(size=15, color="#EC4899", symbol="circle", line=dict(color="#ffffff", width=2)),
                name="我们的模型" if idx == 0 else None,
                showlegend=idx == 0,
                hovertemplate=f"{label}<br>我们的模型：{ours_v:.3f}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="归一化指标值", range=[0, 1], gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="", autorange="reversed")
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=1, xanchor="right", title=None))
    return _base_layout(fig, 430, "多指标占位对比")


def build_domain_ablation_chart(rows: list[dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    labels = df["domain"].replace({"sensor_3band": "传感器域", "source_3band": "源空间"})
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=df["test_auc"],
            mode="markers+lines",
            marker=dict(size=18, color=["#38BDF8", "#8B5CF6"], line=dict(color="#ffffff", width=2)),
            line=dict(color="#7C83FD", width=3),
            customdata=np.stack([df["dims"], df["test_mcc"]], axis=-1),
            hovertemplate="%{x}<br>特征维度：%{customdata[0]}<br>TEST AUC：%{y:.3f}<br>MCC：%{customdata[1]:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8", annotation_text="机会线 0.5")
    fig.update_yaxes(title_text="TEST AUC", range=[0.45, 0.95], gridcolor=GRID_COLOR)
    fig.update_xaxes(title_text="")
    return _base_layout(fig, 380, "同特征跨域消解")
