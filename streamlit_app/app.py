"""
Streamlit dashboard for the SCM + Kernel vs Linear analysis.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(page_title="剖宫产抑郁指标分析", layout="wide")

RESULTS = Path(__file__).resolve().parents[1] / "results"

BASELINE = {
    "A": {"pr_auc": 12 / 18190, "roc_auc": 0.50, "n_pos": 12, "n_total": 18190, "name": "定义A"},
    "B": {"pr_auc": 93 / 18271, "roc_auc": 0.50, "n_pos": 93, "n_total": 18271, "name": "定义B"},
}
METHOD_LABEL = {"linear": "线性", "scm": "SCM+核"}
COUNT_STYLE = {
    1: {"label": "单指标", "color": "#2563eb"},
    2: {"label": "双指标", "color": "#0f766e"},
    3: {"label": "三指标", "color": "#b45309"},
}
TEXT_C = "#334155"
MUTED_C = "#64748b"
GRID_C = "#dce5ef"
PLOT_BG = "#ffffff"
PAPER_BG = "#ffffff"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif;
}
.stApp {
    background: #f5f7fb;
    color: #172033;
}
.block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1 {
    color: #111827 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0 !important;
    margin-bottom: 0.25rem !important;
}
h2, h3 {
    color: #172033 !important;
    font-weight: 650 !important;
    letter-spacing: 0 !important;
}
h3 {
    font-size: 1.15rem !important;
    margin-top: 1.35rem !important;
}
hr {
    border: none;
    border-top: 1px solid #d9e1ec;
    margin: 1.4rem 0;
}
.kicker {
    color: #526173;
    font-size: 0.98rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.summary-card {
    min-height: 126px;
    background: #ffffff;
    border: 1px solid #e1e8f0;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
.summary-label {
    color: #64748b;
    font-size: 0.86rem;
    font-weight: 650;
    margin-bottom: 0.45rem;
}
.summary-value {
    color: #111827;
    font-size: 1.45rem;
    line-height: 1.25;
    font-weight: 720;
    margin-bottom: 0.45rem;
}
.summary-detail {
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.5;
}
.note-panel {
    background: #ffffff;
    border: 1px solid #e1e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    color: #334155;
    font-size: 0.95rem;
    line-height: 1.65;
}
.footer-note {
    color: #64748b;
    font-size: 0.86rem;
    line-height: 1.6;
    text-align: center;
    padding-top: 0.6rem;
}
div[data-testid="stCaptionContainer"] {
    color: #64748b;
}
</style>
""",
    unsafe_allow_html=True,
)


def load_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "scm_kernel_vs_linear_summary.csv")


def fmt_pct(value: float) -> str:
    return f"{value:.3%}"


def fmt_auc(value: float) -> str:
    return f"{value:.4f}"


def compare_wins(df: pd.DataFrame, metric: str) -> tuple[int, int]:
    wide = df.pivot_table(index=["label_type", "indicator_count"], columns="method", values=metric)
    wide = wide.dropna(subset=["linear", "scm"])
    return int((wide["scm"] > wide["linear"]).sum()), int(len(wide))


def best_pr_row(df: pd.DataFrame, label_type: str) -> pd.Series:
    subset = df[df["label_type"] == label_type]
    return subset.sort_values("pr_auc_median", ascending=False).iloc[0]


def render_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
<div class="summary-card">
    <div class="summary-label">{label}</div>
    <div class="summary-value">{value}</div>
    <div class="summary-detail">{detail}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def build_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """
    Two-row comparison chart.

    Bars are overlaid within each method group. Larger bars are drawn first and
    wider; shorter bars are drawn last and narrower so they remain visible.
    """
    group_order = [("A", "linear"), ("A", "scm"), ("B", "linear"), ("B", "scm")]
    group_labels = ["A 线性", "A SCM+核", "B 线性", "B SCM+核"]

    def records_for_metric(metric_key: str) -> list[list[dict]]:
        groups: list[list[dict]] = []
        for label_type, method in group_order:
            bars = []
            for count in [1, 2, 3]:
                subset = df[
                    (df["label_type"] == label_type)
                    & (df["method"] == method)
                    & (df["indicator_count"] == count)
                ]
                if subset.empty:
                    continue
                row = subset.iloc[0]
                raw_value = float(row[f"{metric_key}_median"])
                value = raw_value - BASELINE[label_type][metric_key]
                bars.append(
                    {
                        "value": value,
                        "raw_value": raw_value,
                        "count": count,
                        "features": str(row["best_features"]),
                        "label_type": label_type,
                        "method": method,
                    }
                )
            bars.sort(key=lambda item: abs(item["value"]), reverse=True)
            groups.append(bars)
        return groups

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("PR-AUC 相对阳性率基线", "ROC-AUC 相对 0.50 基线"),
        vertical_spacing=0.16,
    )

    layer_widths = [0.72, 0.52, 0.34]
    metric_rows = [
        ("pr_auc", "PR-AUC", "PR-AUC - 阳性率"),
        ("roc_auc", "ROC-AUC", "ROC-AUC - 0.50"),
    ]

    for row_num, (metric_key, metric_label, axis_title) in enumerate(metric_rows, start=1):
        grouped_records = records_for_metric(metric_key)
        x_pos = list(range(len(group_order)))

        for group_idx, bars in enumerate(grouped_records):
            for layer_idx, item in enumerate(bars):
                style = COUNT_STYLE[item["count"]]
                fig.add_trace(
                    go.Bar(
                        x=[x_pos[group_idx]],
                        y=[item["value"]],
                        width=layer_widths[min(layer_idx, len(layer_widths) - 1)],
                        name=style["label"],
                        legendgroup=style["label"],
                        legendrank=item["count"],
                        showlegend=(row_num == 1 and group_idx == 0),
                        marker=dict(
                            color=style["color"],
                            line=dict(color="#ffffff", width=1.2),
                        ),
                        opacity=0.94,
                        hovertemplate=(
                            f"{group_labels[group_idx]}<br>"
                            f"{style['label']}：{item['features']}<br>"
                            f"{metric_label}：{item['raw_value']:.6f}<br>"
                            "相对基线：%{y:+.6f}<extra></extra>"
                        ),
                    ),
                    row=row_num,
                    col=1,
                )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#94a3b8",
            line_width=1.2,
            row=row_num,
            col=1,
        )
        fig.update_xaxes(
            tickvals=x_pos,
            ticktext=group_labels,
            tickfont=dict(size=13, color=TEXT_C),
            row=row_num,
            col=1,
        )
        fig.update_yaxes(
            title_text=axis_title,
            title_font=dict(size=13, color=TEXT_C),
            tickfont=dict(size=12, color=TEXT_C),
            tickformat=".4f" if metric_key == "pr_auc" else ".2f",
            row=row_num,
            col=1,
        )

    fig.update_layout(
        title=dict(
            text="线性与 SCM+核方法对比",
            font=dict(color="#111827", size=20),
            x=0.01,
        ),
        barmode="overlay",
        bargap=0.42,
        height=720,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_C, size=13),
        legend=dict(
            orientation="h",
            y=1.06,
            x=1,
            xanchor="right",
            title=None,
            font=dict(size=12, color=TEXT_C),
        ),
        margin=dict(l=68, r=28, t=92, b=44),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_color="#172033"),
    )
    fig.update_xaxes(gridcolor=GRID_C, zerolinecolor="#cbd5e1")
    fig.update_yaxes(gridcolor=GRID_C, zerolinecolor="#cbd5e1")
    return fig


def build_pca_chart() -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "低缺失指标 · 定义A",
            "低缺失指标 · 定义B",
            "最佳组合指标 · 定义A",
            "最佳组合指标 · 定义B",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    configs = [
        ("scm_kernel_pca_coords", {"A": (0.354, 0.334), "B": (0.354, 0.334)}),
        ("scm_kernel_pca_best_coords", {"A": (0.338, 0.215), "B": (0.338, 0.215)}),
    ]

    for row_idx, (prefix, ratios) in enumerate(configs):
        for col_idx, label_type in enumerate(["A", "B"]):
            pca_path = RESULTS / f"{prefix}_type{label_type}.csv"
            if not pca_path.exists():
                continue

            coords = pd.read_csv(pca_path)
            non_depressed = coords[coords["depressed"] == 0]
            depressed = coords[coords["depressed"] == 1]
            if len(non_depressed) > 800:
                non_depressed = non_depressed.sample(800, random_state=20260802)

            subplot_row = row_idx + 1
            subplot_col = col_idx + 1
            pc1, pc2 = ratios[label_type]

            fig.add_trace(
                go.Scatter(
                    x=non_depressed["PC1"],
                    y=non_depressed["PC2"],
                    mode="markers",
                    marker=dict(color="rgba(37, 99, 235, 0.24)", size=4),
                    name="非抑郁",
                    showlegend=(row_idx == 0 and col_idx == 0),
                    hovertemplate="非抑郁<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
                ),
                row=subplot_row,
                col=subplot_col,
            )
            fig.add_trace(
                go.Scatter(
                    x=depressed["PC1"],
                    y=depressed["PC2"],
                    mode="markers",
                    marker=dict(
                        color="#e11d48",
                        size=9,
                        symbol="diamond",
                        line=dict(color="#ffffff", width=0.8),
                    ),
                    name="抑郁",
                    showlegend=(row_idx == 0 and col_idx == 0),
                    hovertemplate="抑郁<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
                ),
                row=subplot_row,
                col=subplot_col,
            )
            fig.update_xaxes(
                title_text=f"PC1 ({pc1:.1%})",
                title_font=dict(size=12, color=TEXT_C),
                tickfont=dict(size=11, color=TEXT_C),
                row=subplot_row,
                col=subplot_col,
            )
            fig.update_yaxes(
                title_text=f"PC2 ({pc2:.1%})",
                title_font=dict(size=12, color=TEXT_C),
                tickfont=dict(size=11, color=TEXT_C),
                row=subplot_row,
                col=subplot_col,
            )

    fig.update_layout(
        title=dict(text="PCA 分布概览", font=dict(color="#111827", size=20), x=0.01),
        height=780,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_C, size=12),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", title=None),
        margin=dict(l=58, r=28, t=88, b=46),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_color="#172033"),
    )
    fig.update_xaxes(gridcolor=GRID_C, zerolinecolor="#cbd5e1")
    fig.update_yaxes(gridcolor=GRID_C, zerolinecolor="#cbd5e1")
    return fig


summary = load_summary()
pr_wins, total_pairs = compare_wins(summary, "pr_auc_median")
roc_wins, _ = compare_wins(summary, "roc_auc_median")
best_a = best_pr_row(summary, "A")
best_b = best_pr_row(summary, "B")

st.title("剖宫产围术期指标与产后抑郁识别分析")
st.markdown(
    """
<div class="kicker">
线性逻辑回归与 SCM + RBF 核 SVM 的结果对比。PR-AUC 按阳性率基线解读，ROC-AUC 按 0.50 基线解读。
</div>
""",
    unsafe_allow_html=True,
)

card_a, card_b, card_scm = st.columns(3)
with card_a:
    rate_a = BASELINE["A"]["n_pos"] / BASELINE["A"]["n_total"]
    render_card(
        "定义A · 严格口径",
        f"{BASELINE['A']['n_pos']} / {BASELINE['A']['n_total']:,}",
        (
            f"阳性率 {fmt_pct(rate_a)}。最佳 PR-AUC "
            f"{fmt_auc(best_a['pr_auc_median'])}，{METHOD_LABEL[best_a['method']]}，"
            f"{int(best_a['indicator_count'])} 指标。"
        ),
    )
with card_b:
    rate_b = BASELINE["B"]["n_pos"] / BASELINE["B"]["n_total"]
    render_card(
        "定义B · 宽泛口径",
        f"{BASELINE['B']['n_pos']} / {BASELINE['B']['n_total']:,}",
        (
            f"阳性率 {fmt_pct(rate_b)}。最佳 PR-AUC "
            f"{fmt_auc(best_b['pr_auc_median'])}，{METHOD_LABEL[best_b['method']]}，"
            f"{int(best_b['indicator_count'])} 指标。"
        ),
    )
with card_scm:
    render_card(
        "方法对比",
        f"PR {pr_wins}/{total_pairs}",
        f"SCM 在 PR-AUC 对比中胜出 {pr_wins} 次，ROC-AUC 胜出 {roc_wins} 次；当前主结论以线性模型为准。",
    )

st.markdown("---")
st.subheader("相对基线提升")
st.caption("同一方法组内，较高的柱先绘制在后层，较矮的柱后绘制在前层；颜色表示最佳单、双、三指标组合。")
st.plotly_chart(build_comparison_chart(summary), use_container_width=True, config=PLOTLY_CONFIG)

st.markdown(
    """
<div class="note-panel">
当前结果显示线性模型更稳定。定义A阳性仅12例，定义B阳性93例，PR-AUC 的微小提升也需要结合置信区间谨慎解读。
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")
st.subheader("PCA 分布")
st.caption("蓝点为非抑郁样本抽样展示，红色菱形为抑郁样本。")
st.plotly_chart(build_pca_chart(), use_container_width=True, config=PLOTLY_CONFIG)

st.markdown("---")
st.markdown(
    """
<div class="footer-note">
SCM 实现为 KMeans 压缩多数类 + 边界负样本保留 + 少数类增强 + Nystroem RBF 显式升维；线性基线为 L2 正则化 Logistic Regression。数据来自剖宫产 EHR 导出，姓名字段已哈希脱敏。
</div>
""",
    unsafe_allow_html=True,
)

