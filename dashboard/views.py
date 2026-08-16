from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .charts import (
    build_delta_chart,
    build_domain_ablation_chart,
    build_feature_cloud_chart,
    build_lift_chart,
    build_method_heatmap,
    build_model_position_chart,
    build_pca_chart,
    build_pretraining_ablation_chart,
    build_radar_chart,
    build_version_roadmap_chart,
)
from .components import (
    PAGE_META,
    render_assessment,
    render_evidence_overview,
    render_fusion_blueprint,
    render_glass_note,
    render_hero,
    render_ablation_card_grid,
    render_model_reason_grid,
    render_module_grid,
    render_navigation,
    render_research_journey,
    render_section_header,
)
from .config import BASELINES, JOINT_OUTPUTS, PLOTLY_CONFIG, RESULTS_DIR
from .data import PretrainingData
from .services import (
    build_feature_cloud_terms,
    clinical_threshold_table,
    display_table,
    fmt_auc,
    fmt_pct,
)
from .styles import inject_global_style


def render_app(summary: pd.DataFrame, pretraining: PretrainingData, pretrain_overview: dict[str, Any]) -> None:
    inject_global_style()
    page = render_navigation(_active_page())

    if page == "joint":
        render_joint_analysis()
    elif page == "eeg":
        render_prediction_model(pretraining, pretrain_overview)
    elif page == "physiology":
        render_physiology(summary)
    else:
        render_overview(summary, pretrain_overview)


def _active_page() -> str:
    if "active_section" not in st.session_state:
        st.session_state.active_section = "overview"
    page = st.session_state.active_section
    return page if page in PAGE_META else "overview"


def _best_summary_rows(summary: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    best_a = summary[summary["label_type"] == "A"].sort_values(["pr_auc_median", "roc_auc_median"], ascending=False).iloc[0]
    best_b = summary[summary["label_type"] == "B"].sort_values(["pr_auc_median", "roc_auc_median"], ascending=False).iloc[0]
    return best_a, best_b


def _fpr_one_recall(summary: pd.DataFrame) -> str:
    threshold = clinical_threshold_table(summary)
    if threshold.empty:
        return "FPR=1% 召回率暂无"
    row = threshold[threshold["可承受FPR"] == "1.0%"]
    if row.empty:
        return "FPR=1% 召回率暂无"
    return f"FPR=1% 时当前模型 Recall A/B 为 {row.iloc[0]['当前模型Recall A/B']}"


def _overview_evidence_cards(summary: pd.DataFrame) -> list[dict[str, Any]]:
    best_a, best_b = _best_summary_rows(summary)
    auc_a = float(best_a["pr_auc_median"])
    auc_b = float(best_b["pr_auc_median"])
    max_auc = max(auc_a, auc_b, 1e-12)
    recall_text = _fpr_one_recall(summary).replace("FPR=1% 时当前模型 ", "")
    return [
        {
            "icon": "EV",
            "tag": "Evidence",
            "label": "当前证据构成",
            "value": "生理指标 + 预测模型",
            "bars": [100, 72],
            "tone": "blue",
            "detail": "已完成非影像辅助分析和 EEG 源空间预训练探索，联合 fMRI/EEG 结果等待真实配对数据。",
        },
        {
            "icon": "PR",
            "tag": "PR-AUC",
            "label": "当前最佳分类信号",
            "value": f"A {fmt_auc(auc_a)} / B {fmt_auc(auc_b)}",
            "bars": [auc_a / max_auc * 100, auc_b / max_auc * 100],
            "tone": "cyan",
            "detail": "即使用当前最佳组合，PR-AUC 仍处于很低水平，只能作为辅助证据。",
        },
        {
            "icon": "RD",
            "tag": "Readiness",
            "label": "临床就绪度",
            "value": "Not yet ready",
            "bars": [12, 5],
            "tone": "purple",
            "detail": f"{recall_text}，当前瓶颈是缺少真实配对 fMRI/EEG 数据验证。",
        },
    ]


def _journey_items() -> list[dict[str, str]]:
    return [
        {
            "index": "01",
            "icon": "ECG",
            "title": "Physiological Baseline",
            "subtitle": "生理指标辅助预测",
            "status": "Completed",
            "state": "completed",
            "body": "建立围术期生理指标的非影像基线，明确单独筛查信号的能力上限。",
        },
        {
            "index": "02",
            "icon": "EEG",
            "title": "EEG Representation",
            "subtitle": "预测模型探索",
            "status": "Completed",
            "state": "completed",
            "body": "完成源空间 EEG 图表征和预训练结果展示，为未来融合模型提供候选 encoder。",
        },
        {
            "index": "03",
            "icon": "NET",
            "title": "Multimodal Fusion",
            "subtitle": "fMRI / EEG 联合建模",
            "status": "Pending Data",
            "state": "current",
            "body": "拿到配对数据后，验证 EEG 表征是否提升多模态泛化和解释性。",
        },
        {
            "index": "04",
            "icon": "CLN",
            "title": "Clinical Evaluation",
            "subtitle": "临床工作点评估",
            "status": "To Validate",
            "state": "future",
            "body": "以临床可承受假阳率、召回率和随访负担判断实际可用性。",
        },
    ]


def render_overview(summary: pd.DataFrame, pretrain_overview: dict[str, Any]) -> None:
    render_hero()
    render_section_header(
        "Evidence Overview",
        "当前证据总览",
        "核心问题是当前最好的分类信号有多强、错误水平如何，以及是否已经具备临床判断条件。",
    )
    render_evidence_overview(_overview_evidence_cards(summary))

    render_section_header(
        "Research Journey",
        "图形化研究时间线",
        "从非影像基线到 EEG 表征，再进入 fMRI/EEG 联合验证。当前阶段聚焦多模态融合。",
    )
    render_research_journey(_journey_items())

    heldout = pretrain_overview["risk"].get("heldout_auc")
    heldout_text = f"{heldout:.3f}" if heldout is not None else "暂无"
    render_section_header(
        "Current Assessment",
        "当前判断",
        "框架已形成，但临床可用性尚未成立。关键限制是缺少真实配对数据和稳定的未见人群泛化证据。",
    )
    render_assessment(heldout_text, _fpr_one_recall(summary))


def render_joint_analysis() -> None:
    render_section_header(
        "Joint Analysis",
        "fMRI / EEG 联合分析结果",
        "当前尚未接入真实配对 fMRI/EEG 数据，因此不展示虚构性能。该区域用于承载正式联合模型结果。",
    )
    render_evidence_overview(
        [
            {
                "icon": "DT",
                "tag": "Data",
                "label": "数据状态",
                "value": "待接入",
                "bars": [18],
                "tone": "purple",
                "detail": "需要真实产妇 fMRI/EEG 配对数据和统一标签口径。",
            },
            {
                "icon": "MD",
                "tag": "Model",
                "label": "模型目标",
                "value": "风险判断 + 解释",
                "bars": [62, 44],
                "tone": "blue",
                "detail": "输出个体风险、模态贡献、关键脑区/网络和工作点召回率。",
            },
            {
                "icon": "AB",
                "tag": "Validation",
                "label": "验证设计",
                "value": "预训练增益验证",
                "bars": [54, 31],
                "tone": "green",
                "detail": "比较随机初始化与 EEG encoder 初始化在融合模型中的泛化表现。",
            },
        ]
    )
    render_section_header("Planned Outputs", "计划输出", "真实数据到位后，以下结果位将替换当前占位内容。")
    render_module_grid([{"stage": "待接入", "title": item, "body": "真实数据接入后生成。"} for item in JOINT_OUTPUTS])
    with st.expander("联合建模评估口径", expanded=True):
        st.markdown(
            """
- 数据划分：按受试者划分训练、验证、测试集，避免同一产妇跨集合泄漏。
- 模型对照：fMRI-only、EEG-only、随机初始化融合、EEG预训练初始化融合。
- 主要指标：ROC-AUC、PR-AUC、Sensitivity、Specificity、临床可承受 FPR 下的 Recall。
- 解释输出：EEG图节点/边贡献、fMRI脑区贡献、跨模态一致性。
"""
        )


def render_prediction_model(pretraining: PretrainingData, overview: dict[str, Any]) -> None:
    render_section_header(
        "Prediction Model",
        "预测模型",
        "围绕 EEG 与 fMRI 两个模态建设预训练 encoder，再在真实配对数据上训练融合模型。",
    )
    render_fusion_blueprint(overview["modalities"])

    section = st.segmented_control(
        "预测模型模块",
        ["融合总览", "EEG 预训练", "fMRI 预训练"],
        default="融合总览",
        label_visibility="collapsed",
    )
    if section == "EEG 预训练":
        _render_eeg_pretraining(pretraining, overview)
    elif section == "fMRI 预训练":
        _render_fmri_pretraining()
    else:
        _render_prediction_fusion_overview(overview)


def _render_prediction_fusion_overview(overview: dict[str, Any]) -> None:
    render_section_header(
        "Fusion Overview",
        "EEG 与 fMRI 融合总览",
        "当前 EEG encoder 已完成，fMRI encoder 等待配对数据后训练；融合层将比较单模态、随机初始化和预训练初始化的增益。",
    )
    _metric_grid(_fusion_kpis(overview))
    render_glass_note(overview["risk"]["message"], tone="warning")
    render_section_header(
        "Version Roadmap",
        "版本升级路线图",
        "V0 建立严格评估协议，V1 验证单频带源空间图的局限，V2 通过三频带多视图表征取得当前最佳冻结测试结果。",
    )
    st.plotly_chart(build_version_roadmap_chart(overview["version_roadmap"]), width="stretch", config=PLOTLY_CONFIG)
    render_glass_note(
        "当前最佳效果对应 V2：三频带源空间 PEC 图 + Multi-view GNN，冻结测试 AUC 0.750，MCC 0.491。"
        "该结果接近源空间表征可达上限，但低于传感器域高分对照，需结合批次混杂风险解释。",
        tone="info",
    )
    st.plotly_chart(build_domain_ablation_chart(overview["domain_ablation"]), width="stretch", config=PLOTLY_CONFIG)


def _render_eeg_pretraining(pretraining: PretrainingData, overview: dict[str, Any]) -> None:
    render_section_header(
        "EEG Pretraining",
        "EEG 预训练",
        "已完成从 19 导静息态 EEG 到 Schaefer100 源空间图的建模链条，并完成多频带 GNN 消融与冻结测试评估。",
    )
    st.plotly_chart(build_pretraining_ablation_chart(overview["primary_rows"]), width="stretch", config=PLOTLY_CONFIG)
    render_ablation_card_grid(overview["primary_rows"])

    render_section_header(
        "Benchmarking",
        "现有模型对照",
        "只保留已纳入模型的结果位置和纳入理由，不再展开标准说明卡片。",
    )
    render_model_reason_grid(overview["benchmarks"])
    st.plotly_chart(build_model_position_chart(overview["benchmarks"]), width="stretch", config=PLOTLY_CONFIG)

    with st.expander("EEG 预训练处理链条", expanded=True):
        st.markdown(
            """
- 预处理：19 导静息态 EEG，EC/EO 条件，统一冻结 split 与 subject 级主指标。
- 源定位：Brainstorm/wMNE，对齐 Schaefer100 ROI。
- 连接构图：theta、alpha、beta 三频带 PEC 连接矩阵。
- 当前结论：V2 在冻结测试集上表现最好，但样本量仍小，融合模型需等待真实 fMRI/EEG 配对数据验证。
"""
        )
    _render_pretraining_assets(overview["asset_images"])


def _render_fmri_pretraining() -> None:
    render_section_header(
        "fMRI Pretraining",
        "fMRI 预训练",
        "该模块等待真实配对影像数据后训练；当前只展示将要采用的训练与验证口径，不展示虚构结果。",
    )
    render_module_grid(
        [
            {
                "stage": "待训练",
                "title": "影像表征构建",
                "body": "从脑区时间序列或功能连接图形成 fMRI encoder 输入，并保持受试者级划分。",
            },
            {
                "stage": "待训练",
                "title": "单模态基线",
                "body": "先训练 fMRI-only 模型，明确影像分支独立贡献。",
            },
            {
                "stage": "待训练",
                "title": "融合验证",
                "body": "与 EEG encoder 在共享表示空间融合，比较随机初始化和预训练初始化。",
            },
            {
                "stage": "待训练",
                "title": "临床工作点",
                "body": "按 ROC-AUC、PR-AUC、Sensitivity、Specificity、MCC 和指定 FPR 下 Recall 综合评估。",
            },
        ]
    )


def _benchmark_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[
        ["model", "family", "domain", "reason", "test_auc", "acc", "balacc", "mcc", "params"]
    ].rename(
        columns={
            "model": "模型",
            "family": "类型",
            "domain": "数据域",
            "reason": "选取理由",
            "test_auc": "TEST AUC",
            "acc": "Accuracy",
            "balacc": "Balanced Accuracy",
            "mcc": "MCC",
            "params": "参数量",
        }
    )


def _fusion_kpis(overview: dict[str, Any]) -> list[dict[str, str]]:
    risk = overview["risk"]
    best = overview.get("best_version", {})
    heldout = risk.get("heldout_auc")
    validation = risk.get("validation_auc")
    return [
        {
            "label": "当前最佳验证 AUC",
            "value": f"{validation:.3f}" if validation is not None else "暂无",
            "detail": f"{best.get('version', '暂无')}，频带 {best.get('bands', '暂无')}，融合 {best.get('fusion', '暂无')}",
        },
        {
            "label": "独立 held-out AUC",
            "value": f"{heldout:.3f}" if heldout is not None else "暂无",
            "detail": "这是比交叉验证更接近未见受试者泛化的风险提示指标。",
        },
    ]


def _metric_grid(items: list[dict[str, str]]) -> None:
    cards = []
    tones = ("blue", "cyan", "green")
    for idx, item in enumerate(items):
        cards.append(
            {
                "icon": item["label"][:2].upper(),
                "tag": "Metric",
                "label": item["label"],
                "value": item["value"],
                "bars": [64],
                "tone": tones[idx % len(tones)],
                "detail": item["detail"],
            }
        )
    render_evidence_overview(cards)


def _render_pretraining_assets(asset_images: list[dict[str, Any]]) -> None:
    if not asset_images:
        st.info("未检测到 EEG 预训练结果图。")
        return
    tabs = st.tabs([item["title"] for item in asset_images])
    for tab, item in zip(tabs, asset_images):
        with tab:
            st.image(str(item["path"]), caption=item["caption"], width="stretch")


def render_physiology(summary: pd.DataFrame) -> None:
    render_section_header(
        "Physiology",
        "生理指标辅助分析",
        "围术期生理指标用于评估非影像信息的辅助价值，重点关注最佳分类能力、错误水平和临床工作点距离。",
    )
    best_a, best_b = _best_summary_rows(summary)
    _metric_grid(
        [
            {
                "label": f"{BASELINES['A'].name} · {BASELINES['A'].description}",
                "value": f"{BASELINES['A'].n_pos} / {BASELINES['A'].n_total:,}",
                "detail": f"阳性率 {fmt_pct(BASELINES['A'].pr_auc)}；最佳 PR-AUC {fmt_auc(best_a['pr_auc_median'])}。",
            },
            {
                "label": f"{BASELINES['B'].name} · {BASELINES['B'].description}",
                "value": f"{BASELINES['B'].n_pos} / {BASELINES['B'].n_total:,}",
                "detail": f"阳性率 {fmt_pct(BASELINES['B'].pr_auc)}；最佳 PR-AUC {fmt_auc(best_b['pr_auc_median'])}。",
            },
            {
                "label": "当前分类能力",
                "value": "未达临床线",
                "detail": _fpr_one_recall(summary),
            },
        ]
    )

    tab_signal, tab_error, tab_explain, tab_diagnostic = st.tabs(["分类信号", "错误水平", "信号解释", "方法诊断"])
    with tab_signal:
        terms = build_feature_cloud_terms(summary)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(build_feature_cloud_chart(terms), width="stretch", config=PLOTLY_CONFIG)
        with c2:
            st.plotly_chart(build_radar_chart(summary), width="stretch", config=PLOTLY_CONFIG)
        st.plotly_chart(build_method_heatmap(summary), width="stretch", config=PLOTLY_CONFIG)
    with tab_error:
        render_glass_note(
            "临床达标线按可承受 FPR 和最低召回率倒推。当前生理指标模型未达到常规筛查工作点，"
            "更适合作为多模态模型的背景风险和辅助解释。",
            tone="risk",
        )
        st.dataframe(clinical_threshold_table(summary), width="stretch", hide_index=True)
    with tab_explain:
        st.dataframe(
            display_table(summary),
            width="stretch",
            hide_index=True,
            column_config={
                "PR-AUC": st.column_config.NumberColumn(format="%.6f"),
                "PR提升": st.column_config.NumberColumn(format="%+.6f"),
                "ROC-AUC": st.column_config.NumberColumn(format="%.6f"),
                "ROC提升": st.column_config.NumberColumn(format="%+.6f"),
                "Precision": st.column_config.NumberColumn(format="%.6f"),
            },
        )
        st.plotly_chart(build_pca_chart(RESULTS_DIR), width="stretch", config=PLOTLY_CONFIG)
    with tab_diagnostic:
        render_glass_note("该页签用于核对建模路线差异；正式解读仍以最佳分类能力和错误水平为准。", tone="warning")
        st.plotly_chart(build_lift_chart(summary), width="stretch", config=PLOTLY_CONFIG)
        st.plotly_chart(build_delta_chart(summary), width="stretch", config=PLOTLY_CONFIG)
