from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = APP_ROOT / "results"
EXTERNAL_EEG_ROOT = Path("E:/STUDY_files/EEG_NANDA")
PRETRAIN_ROOT = EXTERNAL_EEG_ROOT if EXTERNAL_EEG_ROOT.exists() else APP_ROOT.parent / "EEG_pretrain"

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

TEXT_COLOR = "#273449"
MUTED_COLOR = "#5f6f86"
GRID_COLOR = "#d8e2ee"


@dataclass(frozen=True)
class BaselineDefinition:
    name: str
    description: str
    n_pos: int
    n_total: int
    roc_auc: float = 0.50

    @property
    def pr_auc(self) -> float:
        return self.n_pos / self.n_total


BASELINES: dict[str, BaselineDefinition] = {
    "A": BaselineDefinition("定义A", "严格口径", 12, 18190),
    "B": BaselineDefinition("定义B", "宽泛口径", 93, 18271),
}

METHOD_STYLE = {
    "linear": {"label": "Linear", "cn": "线性模型", "color": "#2563eb"},
    "scm": {"label": "SCM", "cn": "SCM+核升维", "color": "#e11d48"},
}

COUNT_LABEL = {1: "单指标", 2: "双指标", 3: "三指标"}

SOURCE_IMAGE_CATALOG = [
    (
        "v2_connectivity_7net.png",
        "7网络连接差异",
        "V2 三频带源空间 PEC 图聚合到 7 个功能网络后的组间连接模式。",
    ),
    (
        "v2_node_strength_top24.png",
        "节点强度 Top24",
        "V2 三频带源空间 PEC 图中组间差异较高的 ROI 节点。",
    ),
    (
        "roc_test.png",
        "源空间 ROC",
        "当前最佳 V2 模型在冻结测试集上的 subject-level ROC。",
    ),
    (
        "confusion_test.png",
        "混淆矩阵",
        "当前最佳 V2 模型在冻结测试集上的分类错误类型。",
    ),
    (
        "training_curves.png",
        "训练曲线",
        "当前最佳 V2 模型 5 折验证 AUC 曲线与早停位置。",
    ),
]

CLEAN_SOURCE_IMAGE_CATALOG = [
    (
        "v2_connectivity_7net.png",
        "7网络连接差异",
        "V2 三频带源空间 PEC 聚合到 7 个功能网络后的组间连接模式。",
    ),
    (
        "v2_node_strength_top24.png",
        "节点强度 Top24",
        "V2 三频带源空间 PEC 图中组间差异较高的 ROI 节点。",
    ),
    (
        "roc_test.png",
        "测试集 ROC",
        "当前最佳 V2 模型在冻结测试集上的 subject-level ROC。",
    ),
    (
        "confusion_test.png",
        "测试集混淆矩阵",
        "当前最佳 V2 模型在冻结测试集上的分类结果。",
    ),
    (
        "training_curves.png",
        "训练曲线",
        "当前最佳 V2 模型 5 折验证的 subject-level AUC 曲线。",
    ),
]

JOINT_OUTPUTS = [
    "个体产妇抑郁风险评分",
    "EEG 与 fMRI 模态贡献比例",
    "脑区/脑网络解释",
    "临床可承受 FPR 下的召回率",
    "从随机初始化到预训练初始化的增益",
]

PIPELINE_STEPS = [
    {
        "stage": "已完成",
        "title": "生理指标辅助预测",
        "body": "围术期 EHR 指标已完成 Linear 与 SCM+核升维对比，可作为非影像辅助证据。",
    },
    {
        "stage": "已完成",
        "title": "EEG 预训练探索",
        "body": "已形成 Schaefer100 源空间图、PEC 连接和 GNN encoder 结果图。",
    },
    {
        "stage": "待接入",
        "title": "fMRI/EEG 联合建模",
        "body": "等待真实配对数据后，比较 from-scratch 与 EEG encoder 初始化的融合模型。",
    },
    {
        "stage": "待验证",
        "title": "临床工作点评估",
        "body": "以医院可承受 FPR、召回率和随访负担定义达标线，而不是只看单一 AUC。",
    },
]

VERSION_ROADMAP = [
    {
        "version": "V0",
        "title": "AdaptGNN 复现",
        "input": "alpha PEC",
        "change": "冻结测试集与 subject 级评估协议",
        "dev_auc": 0.673,
        "dev_std": 0.193,
        "test_auc": 0.446,
        "acc": 0.533,
        "balacc": 0.536,
        "mcc": 0.071,
        "params": 8500,
        "status": "历史基线",
    },
    {
        "version": "V1",
        "title": "Alpha Condition-aware GNN",
        "input": "alpha PEC",
        "change": "7网络门控、EC/EO 条件融合、紧凑 GNN",
        "dev_auc": 0.760,
        "dev_std": 0.177,
        "test_auc": 0.405,
        "acc": 0.385,
        "balacc": 0.369,
        "mcc": -0.283,
        "params": 8852,
        "status": "已验证不足",
    },
    {
        "version": "V2",
        "title": "Multi-band Multi-view GNN",
        "input": "theta/alpha/beta PEC",
        "change": "多频带源空间图、band adapter、band attention、EC/EO 均值融合",
        "dev_auc": 0.756,
        "dev_std": 0.143,
        "test_auc": 0.750,
        "acc": 0.733,
        "balacc": 0.741,
        "mcc": 0.491,
        "params": 12085,
        "status": "当前最佳",
    },
]

MODEL_SELECTION_STANDARDS = [
    {
        "criterion": "数据同源",
        "detail": "优先选择在 Mumtaz MDD EEG 数据或其公开版本上报告结果的模型。",
    },
    {
        "criterion": "代码可复核",
        "detail": "优先采用开源实现；无官方代码时只纳入可按论文结构复现且能统一评估协议的模型。",
    },
    {
        "criterion": "任务代表性",
        "detail": "覆盖经典机器学习、深度时序模型、图模型与当前强基准，避免只和单一弱基线比较。",
    },
    {
        "criterion": "协议一致",
        "detail": "本地对照统一使用冻结 split、subject 主指标、final-test-once，不用 epoch 级交叉验证高估结果。",
    },
]

MODEL_BENCHMARKS = [
    {
        "model": "Our V2",
        "family": "源空间图模型",
        "domain": "源空间 Schaefer100",
        "reason": "本项目当前最佳版本，用于标明我们在同口径结果中的位置。",
        "dev_auc": 0.756,
        "dev_std": 0.143,
        "test_auc": 0.750,
        "acc": 0.733,
        "balacc": 0.741,
        "mcc": 0.491,
        "params": 12085,
        "is_ours": True,
    },
    {
        "model": "Mumtaz 2017 LR (spectral features)",
        "family": "经典机器学习",
        "domain": "传感器域 19 导",
        "reason": "原始数据集论文特征，作为传统特征工程基准。",
        "dev_auc": 0.866,
        "dev_std": 0.071,
        "test_auc": 0.875,
        "acc": 0.733,
        "balacc": 0.723,
        "mcc": 0.472,
        "params": 200,
        "is_ours": False,
    },
    {
        "model": "Mumtaz 2018 LR (connectivity features)",
        "family": "经典连接特征",
        "domain": "传感器域 19 导",
        "reason": "原始数据集功能连接论文特征，提供同步似然连接基准。",
        "dev_auc": 0.808,
        "dev_std": 0.134,
        "test_auc": 0.750,
        "acc": 0.667,
        "balacc": 0.652,
        "mcc": 0.342,
        "params": 200,
        "is_ours": False,
    },
    {
        "model": "NeuralBench SimpleConv",
        "family": "开源深度时序模型",
        "domain": "传感器域 19 导",
        "reason": "facebookresearch/neuroai 开源任务架构，代表公开可复核深度基线。",
        "dev_auc": 0.980,
        "dev_std": 0.040,
        "test_auc": 0.893,
        "acc": 0.867,
        "balacc": 0.857,
        "mcc": 0.756,
        "params": 205445,
        "is_ours": False,
    },
    {
        "model": "GCTNet",
        "family": "SOTA 图-Transformer",
        "domain": "传感器域 19 导",
        "reason": "2024 年 EEG 抑郁 SOTA 图-Transformer 代表；无官方代码，本地按论文结构重实现。",
        "dev_auc": 0.840,
        "dev_std": 0.086,
        "test_auc": 0.893,
        "acc": 0.867,
        "balacc": 0.866,
        "mcc": 0.732,
        "params": 37122,
        "is_ours": False,
    },
]
