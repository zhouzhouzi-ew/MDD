"""Streamlit dashboard for updated SCM and Linear comparison."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm


st.set_page_config(page_title="剖宫产抑郁指标分析", layout="wide")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

BASELINE = {
    "A": {"name": "定义A", "desc": "严格口径", "n_pos": 12, "n_total": 18190, "pr_auc": 12 / 18190, "roc_auc": 0.50},
    "B": {"name": "定义B", "desc": "宽泛口径", "n_pos": 93, "n_total": 18271, "pr_auc": 93 / 18271, "roc_auc": 0.50},
}
METHOD_STYLE = {
    "linear": {"label": "Linear", "cn": "线性", "color": "#2563eb"},
    "scm": {"label": "SCM", "cn": "SCM+核升维", "color": "#e11d48"},
}
COUNT_LABEL = {1: "单指标", 2: "双指标", 3: "三指标"}
TEXT_C = "#334155"
GRID_C = "#d8e2ee"


st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif;
}
.stApp { background: #f5f7fb; color: #172033; }
.block-container { max-width: 1320px; padding-top: 1.8rem; padding-bottom: 3rem; }
h1 { color: #111827 !important; font-size: 2rem !important; font-weight: 720 !important; letter-spacing: 0 !important; }
h2, h3 { color: #172033 !important; letter-spacing: 0 !important; }
h3 { font-size: 1.12rem !important; margin-top: 1.25rem !important; }
hr { border: none; border-top: 1px solid #d9e1ec; margin: 1.35rem 0; }
.kicker { color: #526173; font-size: 0.96rem; line-height: 1.6; margin: 0.2rem 0 1rem; }
.metric-card {
    min-height: 128px; background: #fff; border: 1px solid #e1e8f0; border-radius: 8px;
    padding: 0.95rem 1.05rem; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
.metric-label { color: #64748b; font-size: 0.82rem; font-weight: 700; margin-bottom: 0.45rem; }
.metric-value { color: #111827; font-size: 1.38rem; line-height: 1.25; font-weight: 760; margin-bottom: 0.42rem; }
.metric-detail { color: #475569; font-size: 0.88rem; line-height: 1.48; }
.note-panel {
    background: #fff; border: 1px solid #e1e8f0; border-left: 4px solid #0f766e;
    border-radius: 8px; padding: 0.85rem 1rem; color: #334155; font-size: 0.94rem; line-height: 1.65;
}
.method-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.85rem; }
.method-card { background: #fff; border: 1px solid #e1e8f0; border-radius: 8px; padding: 0.9rem 1rem; }
.method-title { color: #111827; font-size: 0.98rem; font-weight: 750; margin-bottom: 0.35rem; }
.method-body { color: #475569; font-size: 0.88rem; line-height: 1.55; }
.footer-note { color: #64748b; font-size: 0.84rem; line-height: 1.6; text-align: center; padding-top: 0.5rem; }
div[data-testid="stCaptionContainer"] { color: #64748b; }
</style>
""",
    unsafe_allow_html=True,
)


def load_summary() -> pd.DataFrame:
    path = RESULTS / "scm_kernel_vs_linear_summary.csv"
    if not path.exists():
        st.error(f"未找到结果文件：{path}")
        st.stop()
    df = pd.read_csv(path)
    df["indicator_count"] = df["indicator_count"].astype(int)
    df["method_label"] = df["method"].map(lambda m: METHOD_STYLE.get(m, {"cn": m})["cn"])
    df["label_name"] = df["label_type"].map(lambda v: f"{BASELINE[v]['name']} · {BASELINE[v]['desc']}")
    df["count_label"] = df["indicator_count"].map(COUNT_LABEL)
    df["x_label"] = df["label_type"] + " · " + df["count_label"]
    df["pr_lift"] = df.apply(lambda r: float(r["pr_auc_median"]) - BASELINE[r["label_type"]]["pr_auc"], axis=1)
    df["roc_lift"] = df["roc_auc_median"].astype(float) - 0.50
    return df


def fmt_pct(value: float) -> str:
    return f"{value:.3%}"


def fmt_auc(value: float) -> str:
    return f"{value:.4f}"


def fmt_signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value}</div>
  <div class="metric-detail">{detail}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def best_row(df: pd.DataFrame, label_type: str) -> pd.Series:
    return df[df["label_type"] == label_type].sort_values(["pr_auc_median", "roc_auc_median"], ascending=False).iloc[0]


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
    wide = df.pivot_table(index=["label_type", "indicator_count"], columns="method", values=["pr_auc_median", "roc_auc_median"], aggfunc="first")
    if not {"linear", "scm"}.issubset(set(wide["pr_auc_median"].columns)):
        return pd.DataFrame()
    out = pd.DataFrame(index=wide.index).reset_index()
    out["x_label"] = out["label_type"] + " · " + out["indicator_count"].map(COUNT_LABEL)
    out["pr_delta"] = wide["pr_auc_median"]["scm"].to_numpy() - wide["pr_auc_median"]["linear"].to_numpy()
    out["roc_delta"] = wide["roc_auc_median"]["scm"].to_numpy() - wide["roc_auc_median"]["linear"].to_numpy()
    return out


def build_lift_chart(df: pd.DataFrame) -> go.Figure:
    x_order = [f"{lt} · {COUNT_LABEL[c]}" for lt in ["A", "B"] for c in [1, 2, 3]]
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("PR-AUC 相对阳性率基线", "ROC-AUC 相对 0.50 基线"),
        vertical_spacing=0.15,
    )
    for row, metric, raw_col, title in [
        (1, "pr_lift", "pr_auc_median", "PR-AUC"),
        (2, "roc_lift", "roc_auc_median", "ROC-AUC"),
    ]:
        for method in [m for m in ["linear", "scm"] if m in set(df["method"] )]:
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
        fig.update_yaxes(title_text="提升量", tickfont=dict(size=11, color=TEXT_C), gridcolor=GRID_C, row=row, col=1)
        fig.update_xaxes(tickfont=dict(size=12, color=TEXT_C), row=row, col=1)

    fig.update_layout(
        title=dict(text="各方法最佳组合的相对基线提升", x=0.01, font=dict(size=20, color="#111827")),
        barmode="group",
        bargap=0.28,
        height=700,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color=TEXT_C, size=12),
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right", title=None),
        margin=dict(l=60, r=28, t=92, b=52),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_color="#172033"),
    )
    return fig


def build_delta_chart(df: pd.DataFrame) -> go.Figure:
    deltas = delta_table(df)
    if deltas.empty:
        return go.Figure()
    colors_pr = np.where(deltas["pr_delta"] >= 0, "#0f766e", "#b91c1c")
    colors_roc = np.where(deltas["roc_delta"] >= 0, "#0f766e", "#b91c1c")
    fig = make_subplots(rows=2, cols=1, subplot_titles=("PR-AUC 差值：SCM - Linear", "ROC-AUC 差值：SCM - Linear"), vertical_spacing=0.17)
    fig.add_trace(go.Bar(x=deltas["x_label"], y=deltas["pr_delta"], marker_color=colors_pr, hovertemplate="%{x}<br>差值：%{y:+.6f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=deltas["x_label"], y=deltas["roc_delta"], marker_color=colors_roc, hovertemplate="%{x}<br>差值：%{y:+.6f}<extra></extra>"), row=2, col=1)
    for row in [1, 2]:
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1.1, row=row, col=1)
        fig.update_xaxes(tickfont=dict(size=12, color=TEXT_C), row=row, col=1)
        fig.update_yaxes(title_text="SCM - Linear", gridcolor=GRID_C, tickfont=dict(size=11, color=TEXT_C), row=row, col=1)
    fig.update_layout(
        title=dict(text="更新后 SCM 与 Linear 的直接差值", x=0.01, font=dict(size=20, color="#111827")),
        showlegend=False,
        height=560,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=66, r=28, t=88, b=48),
        font=dict(color=TEXT_C, size=12),
    )
    return fig


def pca_feature_text(kind: str, label_type: str) -> str:
    prefix = "low_missing" if kind == "low" else "best"
    path = RESULTS / f"scm_kernel_pca_{prefix}_type{label_type}_loadings.csv"
    if not path.exists() and kind == "low":
        return "年龄 + BMI + 术前-肌酐"
    if not path.exists():
        return "未找到指标文件"
    features = pd.read_csv(path)["feature"].astype(str).tolist()
    return " + ".join(features)


def pca_title(kind_label: str, label_type: str, features: str) -> str:
    return f"{kind_label} · 定义{label_type}<br><sup>{features}</sup>"

def build_pca_chart() -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            pca_title("低缺失指标", "A", pca_feature_text("low", "A")),
            pca_title("低缺失指标", "B", pca_feature_text("low", "B")),
            pca_title("最佳组合指标", "A", pca_feature_text("best", "A")),
            pca_title("最佳组合指标", "B", pca_feature_text("best", "B")),
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
            path = RESULTS / f"{prefix}_type{label_type}.csv"
            if not path.exists():
                continue
            coords = pd.read_csv(path)
            non_dep = coords[coords["depressed"] == 0]
            dep = coords[coords["depressed"] == 1]
            if len(non_dep) > 800:
                non_dep = non_dep.sample(800, random_state=20260802)
            r, c = row_idx + 1, col_idx + 1
            pc1, pc2 = ratios[label_type]
            fig.add_trace(go.Scatter(x=non_dep["PC1"], y=non_dep["PC2"], mode="markers", marker=dict(color="rgba(37, 99, 235, 0.24)", size=4), name="非抑郁", showlegend=(row_idx == 0 and col_idx == 0), hovertemplate="非抑郁<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>"), row=r, col=c)
            fig.add_trace(go.Scatter(x=dep["PC1"], y=dep["PC2"], mode="markers", marker=dict(color="#e11d48", size=9, symbol="diamond", line=dict(color="#ffffff", width=0.8)), name="抑郁", showlegend=(row_idx == 0 and col_idx == 0), hovertemplate="抑郁<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>"), row=r, col=c)
            fig.update_xaxes(title_text=f"PC1 ({pc1:.1%})", title_font=dict(size=12, color=TEXT_C), tickfont=dict(size=11, color=TEXT_C), gridcolor=GRID_C, row=r, col=c)
            fig.update_yaxes(title_text=f"PC2 ({pc2:.1%})", title_font=dict(size=12, color=TEXT_C), tickfont=dict(size=11, color=TEXT_C), gridcolor=GRID_C, row=r, col=c)
    fig.update_layout(title=dict(text="PCA 分布概览", x=0.01, font=dict(color="#111827", size=20)), height=780, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color=TEXT_C, size=12), legend=dict(orientation="h", y=1.05, x=1, xanchor="right", title=None), margin=dict(l=58, r=28, t=88, b=46))
    return fig


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["label_name", "count_label", "method_label", "best_features", "pr_auc_median", "pr_lift", "roc_auc_median", "roc_lift", "precision_median"]].copy()
    out = out.rename(columns={
        "label_name": "定义",
        "count_label": "指标数",
        "method_label": "方法",
        "best_features": "最佳指标组合",
        "pr_auc_median": "PR-AUC",
        "pr_lift": "PR提升",
        "roc_auc_median": "ROC-AUC",
        "roc_lift": "ROC提升",
        "precision_median": "Precision",
    })
    return out


summary = load_summary()
mtime = pd.Timestamp((RESULTS / "scm_kernel_vs_linear_summary.csv").stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
pr_scm, pr_linear, pr_tie, pr_total = method_outcomes(summary, "pr_auc_median")
roc_scm, roc_linear, roc_tie, roc_total = method_outcomes(summary, "roc_auc_median")
best_a = best_row(summary, "A")
best_b = best_row(summary, "B")

st.title("剖宫产围术期指标与产后抑郁识别分析")
st.markdown(
    f"""
<div class="kicker">
更新后 Linear 与 SCM+核升维方法的结果页。PR-AUC 按阳性率基线解读，ROC-AUC 按 0.50 基线解读。结果文件更新时间：{mtime}。
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    rate = BASELINE["A"]["n_pos"] / BASELINE["A"]["n_total"]
    card("定义A · 严格口径", f"{BASELINE['A']['n_pos']} / {BASELINE['A']['n_total']:,}", f"阳性率 {fmt_pct(rate)}。最佳 PR-AUC {fmt_auc(best_a['pr_auc_median'])}，{best_a['method_label']}，{COUNT_LABEL[int(best_a['indicator_count'])]}。")
with c2:
    rate = BASELINE["B"]["n_pos"] / BASELINE["B"]["n_total"]
    card("定义B · 宽泛口径", f"{BASELINE['B']['n_pos']} / {BASELINE['B']['n_total']:,}", f"阳性率 {fmt_pct(rate)}。最佳 PR-AUC {fmt_auc(best_b['pr_auc_median'])}，{best_b['method_label']}，{COUNT_LABEL[int(best_b['indicator_count'])]}。")
with c3:
    card("SCM vs Linear", f"PR {pr_scm}/{pr_total}", f"SCM 在 PR-AUC 胜出 {pr_scm} 次、Linear 胜出 {pr_linear} 次、持平 {pr_tie} 次；ROC-AUC 中 SCM 胜出 {roc_scm} 次。")

st.markdown("---")
st.subheader("相对基线提升")
st.caption("柱形为每个定义、每个指标数下的最佳组合表现。PR-AUC 的随机基线为阳性率，ROC-AUC 的随机基线为 0.50。")
st.plotly_chart(build_lift_chart(summary), width='stretch', config=PLOTLY_CONFIG)

st.subheader("SCM 与 Linear 的差值")
st.caption("绿色表示 SCM 高于 Linear，红色表示 SCM 低于 Linear。")
st.plotly_chart(build_delta_chart(summary), width='stretch', config=PLOTLY_CONFIG)

st.markdown(
    """
<div class="note-panel">
当前新版 SCM 已包含训练折内预处理、KMeans 多数类压缩、边界负样本保留、少数类增强、Nystroem RBF 显式升维和线性锚定。结果显示：SCM 与 Linear 的差距明显收敛，但在当前数据上 PR-AUC 主结论仍更偏向 Linear；Type A 阳性数仅 12，应按探索性结果解读。
</div>
""",
    unsafe_allow_html=True,
)

# ---------- 临床达标线参考:由"可承受 FPR + 最低召回率"倒推合格线 ----------
RECALL_REQ = 0.70  # 筛查默认最低召回率


def _qualified_roc(fpr: float, recall: float) -> float:
    d = norm.ppf(1 - fpr) - norm.ppf(1 - recall)
    return float(norm.cdf(d / np.sqrt(2)))


def _qualified_pr(fpr: float, recall: float, pi: float, n: int = 120001) -> float:
    d = norm.ppf(1 - fpr) - norm.ppf(1 - recall)
    t = np.linspace(-8.0, d + 8.0, n)
    tpr = 1.0 - norm.cdf(t - d)
    fprv = 1.0 - norm.cdf(t)
    den = pi * tpr + (1.0 - pi) * fprv
    prec = np.divide(pi * tpr, den, out=np.ones_like(den), where=den > 0)
    return float(np.trapezoid(prec[::-1], tpr[::-1]))


def _precision_at(fpr: float, recall: float, pi: float) -> float:
    return pi * recall / (pi * recall + (1.0 - pi) * fpr)


def _recall_at(fpr: float, roc_auc: float) -> float:
    d = np.sqrt(2) * norm.ppf(roc_auc)
    return 1.0 - norm.cdf(norm.ppf(1 - fpr) - d)


pi_a = BASELINE["A"]["n_pos"] / BASELINE["A"]["n_total"]
pi_b = BASELINE["B"]["n_pos"] / BASELINE["B"]["n_total"]
qual_rows = []
for fpr in (0.005, 0.01, 0.05):
    qual_rows.append({
        "可承受FPR": f"{fpr:.1%}",
        "达标ROC-AUC": f"{_qualified_roc(fpr, RECALL_REQ):.3f}",
        "达标PR-AUC A/B": f"{_qualified_pr(fpr, RECALL_REQ, pi_a):.3f} / {_qualified_pr(fpr, RECALL_REQ, pi_b):.3f}",
        "工作点Precision A/B": f"{_precision_at(fpr, RECALL_REQ, pi_a):.1%} / {_precision_at(fpr, RECALL_REQ, pi_b):.1%}",
        "每例真阳需随访假阳 A/B": f"{(1 - pi_a) * fpr / (pi_a * RECALL_REQ):.1f} / {(1 - pi_b) * fpr / (pi_b * RECALL_REQ):.1f}",
        "当前模型Recall A/B": f"{_recall_at(fpr, best_a['roc_auc_median']):.0%} / {_recall_at(fpr, best_b['roc_auc_median']):.0%}",
    })
qual_df = pd.DataFrame(qual_rows)

st.markdown("---")
st.subheader("临床达标线参考")
st.caption("判定逻辑:由临床可承受的假阳性率(FPR,愿意标记复核的人群比例)+ 最低召回率(筛查默认 70%)倒推合格线,而非给定固定 AUC。")
st.dataframe(qual_df, width='stretch', hide_index=True)
st.markdown(
    f"""
<div class="note-panel">
<b>解读</b>:按 FPR=1% 的常规随访负担,合格要求是召回率≥70% 时 ROC-AUC ≥ {_qualified_roc(0.01, RECALL_REQ):.3f}。
当前最优模型(Linear)在 FPR=1% 下实际召回率:定义A ≈ {_recall_at(0.01, best_a['roc_auc_median']):.0%}、定义B ≈ {_recall_at(0.01, best_b['roc_auc_median']):.0%},未达临床线。<br>
定义A 全组仅 12 例阳性,受患病率天花板限制,Precision 与 PR-AUC 天然偏低,应以"该 FPR 下的召回率 / 假阳随访负担"判定,不宜用 PR-AUC 卡线。
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")
st.subheader("最佳组合明细")
st.dataframe(
    display_table(summary),
    width='stretch',
    hide_index=True,
    column_config={
        "PR-AUC": st.column_config.NumberColumn(format="%.6f"),
        "PR提升": st.column_config.NumberColumn(format="%+.6f"),
        "ROC-AUC": st.column_config.NumberColumn(format="%.6f"),
        "ROC提升": st.column_config.NumberColumn(format="%+.6f"),
        "Precision": st.column_config.NumberColumn(format="%.6f"),
    },
)

st.markdown("---")
st.subheader("PCA 分布")
st.caption("蓝点为非抑郁样本抽样展示，红色菱形为抑郁样本。")
st.markdown(
    f"""
<div class="note-panel">
低缺失指标：{pca_feature_text("low", "A")}。<br>
最佳组合指标 · 定义A：{pca_feature_text("best", "A")}。<br>
最佳组合指标 · 定义B：{pca_feature_text("best", "B")}。
</div>
""",
    unsafe_allow_html=True,
)
st.plotly_chart(build_pca_chart(), width='stretch', config=PLOTLY_CONFIG)

st.markdown("---")
st.markdown(
    """
<div class="method-grid">
  <div class="method-card">
    <div class="method-title">Linear</div>
    <div class="method-body">Median imputation + RobustScaler + L2 Logistic Regression，使用 class_weight="balanced" 处理类别不平衡。</div>
  </div>
  <div class="method-card">
    <div class="method-title">SCM+核升维</div>
    <div class="method-body">KMeans 压缩多数类，保留边界负样本，训练折内少数类增强，Nystroem RBF 显式升维，并加入线性锚定稳定极少阳性场景。</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="footer-note">
数据来自剖宫产 EHR 导出；姓名字段已哈希脱敏。
</div>
""",
    unsafe_allow_html=True,
)

