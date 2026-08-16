from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


PAGE_META = {
    "overview": {"label": "总览", "en": "Overview", "icon": ":material/home:"},
    "joint": {"label": "联合分析", "en": "Joint", "icon": ":material/hub:"},
    "eeg": {"label": "预测模型", "en": "Model", "icon": ":material/neurology:"},
    "physiology": {"label": "生理指标", "en": "Physiology", "icon": ":material/monitor_heart:"},
}


def render_navigation(active_page: str) -> str:
    pending_section = st.session_state.pop("pending_section", None)
    if pending_section in PAGE_META:
        active_page = pending_section
        st.session_state.active_section = pending_section

    st.sidebar.markdown(
        """
<div class="nav-rail">
  <div class="nav-orb"></div>
  <div class="nav-brand">
    <div class="brand-mark">NE</div>
    <div class="brand-copy">
      <strong>NeuroEval</strong>
      <span>Maternal Neuro Research</span>
    </div>
  </div>
  <div class="nav-helper">Sections</div>
</div>
""",
        unsafe_allow_html=True,
    )
    selected = active_page
    for key, meta in PAGE_META.items():
        active = key == active_page
        label = f"{meta['label']}  ·  {meta['en']}"
        if st.sidebar.button(
            label,
            key=f"nav_{key}",
            type="primary" if active else "secondary",
            icon=meta["icon"],
            width="stretch",
        ):
            st.session_state.active_section = key
            st.session_state.pending_section = key
            st.rerun()
    st.sidebar.markdown('<div class="nav-collapse-dot"></div>', unsafe_allow_html=True)
    return selected


def render_hero() -> None:
    st.markdown(
        """
<section class="hero-section reveal">
  <div class="hero-content">
    <div class="eyebrow">Research Stage&nbsp;&nbsp;<span>Multimodal Validation</span></div>
    <h1>基于 fMRI 和 EEG 的产妇抑郁判断</h1>
    <p class="hero-subtitle">Multimodal Neuroimaging &amp; EEG Depression Assessment</p>
    <p class="hero-body">
      融合生理指标、EEG 表征与未来 fMRI 数据，形成可解释的多模态产妇抑郁辅助研究框架。
    </p>
    <div class="hero-actions">
      <span class="stage-pill">Paired data pending</span>
      <span class="stage-pill muted">Clinical endpoint driven</span>
    </div>
  </div>
  <div class="hero-visual" aria-hidden="true">
    <svg class="pipeline-illustration" viewBox="0 0 520 360" role="img">
      <defs>
        <linearGradient id="nodeBlue" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#EAF4FF" />
          <stop offset="100%" stop-color="#F4F0FF" />
        </linearGradient>
        <linearGradient id="coreGrad" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#3B82F6" />
          <stop offset="100%" stop-color="#7C83FD" />
        </linearGradient>
      </defs>
      <path class="pipeline-link" d="M168 92C213 94 227 132 260 156" />
      <path class="pipeline-link" d="M168 180H252" />
      <path class="pipeline-link" d="M168 268C214 264 229 222 260 204" />
      <path class="pipeline-link active" d="M322 180H410" />
      <g class="source-node physiology">
        <rect x="52" y="54" width="116" height="76" rx="22" />
        <text x="78" y="84">PHY</text>
        <path class="signal-wave" d="M74 104h15l8-18 13 38 12-24h26" />
        <text class="node-label" x="80" y="121">生理指标</text>
      </g>
      <g class="source-node eeg">
        <rect x="52" y="142" width="116" height="76" rx="22" />
        <text x="79" y="172">EEG</text>
        <path class="signal-wave" d="M72 193c9 0 7-15 16-15s7 15 16 15 7-15 16-15 7 15 16 15" />
        <text class="node-label" x="84" y="210">脑电表征</text>
      </g>
      <g class="source-node fmri">
        <rect x="52" y="230" width="116" height="76" rx="22" />
        <text x="78" y="260">fMRI</text>
        <ellipse cx="117" cy="282" rx="26" ry="13" />
        <path d="M117 269v26M93 282h48" />
        <text class="node-label" x="85" y="298">影像数据</text>
      </g>
      <g class="fusion-core">
        <circle cx="292" cy="180" r="54" />
        <circle cx="292" cy="180" r="31" />
        <path d="M258 180h68M292 146v68M270 160c14 14 31 14 45 0M270 200c14-14 31-14 45 0" />
        <text x="270" y="186">Fusion</text>
      </g>
      <g class="risk-output">
        <rect x="410" y="122" width="78" height="116" rx="26" />
        <path d="M436 158h26M449 145v26M432 198h34" />
        <text x="427" y="218">Risk</text>
      </g>
      <g class="visual-crosses">
        <path d="M214 72h18M223 63v18M454 82h18M463 73v18M222 286h18M231 277v18" />
        <rect x="206" y="122" width="10" height="10" rx="2" />
        <rect x="368" y="244" width="8" height="8" rx="2" />
      </g>
    </svg>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_section_header(kicker: str, title: str, body: str = "") -> None:
    st.markdown(
        f"""
<div class="section-heading reveal">
  <span>{escape(kicker)}</span>
  <h2>{escape(title)}</h2>
  <p>{escape(body)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_evidence_overview(cards: list[dict[str, Any]]) -> None:
    tones = ("blue", "cyan", "purple", "green")
    blocks = []
    for idx, card in enumerate(cards):
        tone = card.get("tone", tones[idx % len(tones)])
        bars = "".join(
            f'<span style="width:{max(4, min(100, float(width))):.1f}%"></span>'
            for width in card.get("bars", [])
        )
        blocks.append(
            f"""
<article class="evidence-card tone-{escape(str(tone))} reveal" style="--delay:{idx * 80}ms">
  <div class="card-top">
    <span class="card-icon">{escape(card["icon"])}</span>
    <span class="card-tag">{escape(card["tag"])}</span>
  </div>
  <div class="card-label">{escape(card["label"])}</div>
  <div class="card-number">{escape(card["value"])}</div>
  <div class="mini-bars">{bars}</div>
  <p>{escape(card["detail"])}</p>
</article>
"""
        )
    st.markdown(f'<div class="evidence-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)


def render_research_journey(items: list[dict[str, str]]) -> None:
    blocks = []
    for idx, item in enumerate(items):
        state = item.get("state", "future")
        blocks.append(
            f"""
<article class="journey-node {escape(state)} reveal" style="--delay:{idx * 90}ms">
  <div class="journey-point"><span>{escape(item["icon"])}</span></div>
  <div class="journey-card">
    <div class="journey-index">{escape(item["index"])}</div>
    <h3>{escape(item["title"])}</h3>
    <strong>{escape(item["subtitle"])}</strong>
    <p>{escape(item["body"])}</p>
    <em class="status-badge">{escape(item["status"])}</em>
  </div>
</article>
"""
        )
    st.markdown(
        f"""
<div class="research-journey">
  <div class="journey-line"></div>
  {"".join(blocks)}
</div>
""",
        unsafe_allow_html=True,
    )


def render_assessment(heldout_auc: str, recall_text: str) -> None:
    st.markdown(
        f"""
<section class="assessment-panel reveal">
  <div class="assessment-orbit">
    <div class="progress-ring"><span>3 / 4</span></div>
    <p>Research stages established</p>
  </div>
  <div class="assessment-copy">
    <span class="assessment-kicker">Current Assessment</span>
    <h2>研究框架已建立，临床分类能力仍需真实 fMRI / EEG 数据验证</h2>
    <div class="assessment-facts">
      <div><small>EEG held-out AUC</small><strong>{escape(heldout_auc)}</strong></div>
      <div><small>Current Bottleneck</small><strong>Paired fMRI / EEG data unavailable</strong></div>
      <div><small>Clinical Readiness</small><strong class="risk-text">Not yet ready</strong></div>
    </div>
    <p>
      生理指标模型即使用当前最佳组合，在常规工作点下召回率仍不足；{escape(recall_text)}。
      当前证据链支持继续进入多模态验证阶段，但尚不足以形成可部署的临床筛查系统。
    </p>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_glass_note(text: str, tone: str = "info") -> None:
    st.markdown(f'<div class="glass-note {escape(tone)}">{text}</div>', unsafe_allow_html=True)


def render_fusion_blueprint(modalities: dict[str, Any]) -> None:
    eeg = modalities.get("eeg", {})
    fmri = modalities.get("fmri", {})
    st.markdown(
        f"""
<section class="fusion-blueprint reveal">
  <div class="fusion-lane eeg-lane">
    <span class="lane-status done">{escape(str(eeg.get("status", "已完成")))}</span>
    <h3>EEG Encoder</h3>
    <p>{escape(str(eeg.get("summary", "源空间图表征已完成。")))}</p>
    <div class="lane-stack">
      <span>三频带 PEC</span>
      <span>Schaefer100 ROI</span>
      <span>GNN 表征</span>
    </div>
  </div>
  <div class="fusion-core-panel">
    <div class="fusion-ring">
      <span>Fusion</span>
      <small>shared latent space</small>
    </div>
    <div class="fusion-arrows">
      <i></i><i></i>
    </div>
    <div class="fusion-output">
      <strong>风险评分</strong>
      <strong>模态贡献</strong>
      <strong>脑区解释</strong>
    </div>
  </div>
  <div class="fusion-lane fmri-lane pending">
    <span class="lane-status wait">{escape(str(fmri.get("status", "待训练")))}</span>
    <h3>fMRI Encoder</h3>
    <p>{escape(str(fmri.get("summary", "等待真实配对数据。")))}</p>
    <div class="lane-stack">
      <span>脑区时间序列</span>
      <span>功能连接图</span>
      <span>影像表征</span>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_module_grid(items: list[dict[str, str]]) -> None:
    blocks = []
    for item in items:
        status = "done" if item["stage"] == "已完成" else "pending"
        blocks.append(
            f"""
<div class="module-card {status}">
  <span>{escape(item["stage"])}</span>
  <h3>{escape(item["title"])}</h3>
  <p>{escape(item["body"])}</p>
</div>
"""
        )
    st.markdown(f'<div class="module-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)


def render_model_reason_grid(models: list[dict[str, Any]]) -> None:
    blocks = []
    for idx, model in enumerate(models):
        tone = "ours" if model.get("is_ours") else "reference"
        role = "本项目" if model.get("is_ours") else "对照模型"
        blocks.append(
            f"""
<article class="model-reason-card {tone} reveal" style="--delay:{idx * 70}ms">
  <div class="model-reason-top">
    <span>{escape(role)}</span>
    <small>{escape(str(model.get("domain", "")))}</small>
  </div>
  <h3>{escape(str(model.get("model", "")))}</h3>
  <strong>{escape(str(model.get("family", "")))}</strong>
  <p>{escape(str(model.get("reason", "")))}</p>
</article>
"""
        )
    st.markdown(f'<div class="model-reason-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)


def render_ablation_card_grid(rows: list[dict[str, Any]]) -> None:
    blocks = []
    for idx, row in enumerate(rows[:4]):
        auc = row.get("val_auc")
        test_auc = row.get("test_auc")
        auc_text = f"{float(auc):.3f}" if auc is not None else "暂无"
        test_text = f"{float(test_auc):.3f}" if test_auc is not None else "锁定后评估"
        tags = [
            str(row.get("bands", "")).replace(",", " + "),
            f"SSL {float(row.get('ssl') or 0):.2f}",
            f"{row.get('params') or '-'} params",
        ]
        tag_html = "".join(f"<span>{escape(tag)}</span>" for tag in tags if tag.strip())
        blocks.append(
            f"""
<article class="ablation-card reveal" style="--delay:{idx * 70}ms">
  <div class="ablation-rank">#{idx + 1}</div>
  <h3>{escape(str(row.get("run", "")).replace("_", " "))}</h3>
  <div class="ablation-score">
    <div><small>DEV AUC</small><strong>{escape(auc_text)}</strong></div>
    <div><small>TEST AUC</small><strong>{escape(test_text)}</strong></div>
  </div>
  <div class="ablation-tags">{tag_html}</div>
</article>
"""
        )
    st.markdown(f'<div class="ablation-card-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)
