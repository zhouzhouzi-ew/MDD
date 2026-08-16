from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UiContractTests(unittest.TestCase):
    def test_overview_does_not_surface_method_win_loss(self):
        source = (ROOT / "dashboard" / "views.py").read_text(encoding="utf-8")

        overview_start = source.index("def render_overview")
        joint_start = source.index("def render_joint_analysis")
        overview_source = source[overview_start:joint_start]

        self.assertNotIn("overview_cards(", overview_source)
        self.assertNotIn("SCM vs Linear", overview_source)
        self.assertIn("Current Assessment", overview_source)
        self.assertIn("Research Journey", overview_source)

    def test_visual_system_css_contract(self):
        source = (ROOT / "dashboard" / "styles.py").read_text(encoding="utf-8")
        css_dir = ROOT / "dashboard" / "assets"
        if css_dir.exists():
            for css_path in sorted(css_dir.glob("*.css")):
                source += "\n" + css_path.read_text(encoding="utf-8")

        for token in ("--bg", "--surface", "--primary", "--secondary", "--risk", "--radius-lg"):
            self.assertIn(token, source)
        for selector in (
            ".nav-rail",
            ".hero-visual",
            ".research-journey",
            ".assessment-panel",
            ".evidence-card.tone-blue",
            ".journey-node.completed",
            ".journey-node.current",
            ".journey-node.future",
        ):
            self.assertIn(selector, source)
        self.assertIn("prefers-reduced-motion", source)

    def test_navigation_uses_session_state_not_query_pages(self):
        views = (ROOT / "dashboard" / "views.py").read_text(encoding="utf-8")
        components = (ROOT / "dashboard" / "components.py").read_text(encoding="utf-8")
        combined = views + "\n" + components

        self.assertIn("active_section", combined)
        self.assertIn("st.sidebar.button", combined)
        self.assertIn("pending_section", components)
        self.assertIn("st.rerun()", components)
        self.assertNotIn("use_container_width=True", combined)
        self.assertIn(":material/home:", components)
        self.assertIn("icon=meta[\"icon\"]", components)
        self.assertNotIn("st.query_params", combined)
        self.assertNotIn("href=\"?page=", combined)

    def test_prediction_model_page_contract(self):
        views = (ROOT / "dashboard" / "views.py").read_text(encoding="utf-8")
        components = (ROOT / "dashboard" / "components.py").read_text(encoding="utf-8")
        charts = (ROOT / "dashboard" / "charts.py").read_text(encoding="utf-8")
        config = (ROOT / "dashboard" / "config.py").read_text(encoding="utf-8")

        self.assertIn('"eeg": {"label": "预测模型"', components)
        self.assertIn("render_prediction_model", views)
        self.assertIn("render_fusion_blueprint", components)
        self.assertIn("st.segmented_control", views)
        self.assertIn("EEG 预训练", views)
        self.assertIn("fMRI 预训练", views)
        self.assertIn("待训练", views)
        self.assertIn("版本升级路线图", views)
        self.assertIn("现有模型对照", views)
        self.assertIn("build_model_position_chart", charts)
        self.assertIn("MODEL_BENCHMARKS", config)

        combined = "\n".join([views, components, charts, config])
        for phrase in ("路线A", "路线B"):
            self.assertNotIn(phrase, combined)

    def test_fusion_overview_does_not_render_eeg_asset_metric(self):
        views = (ROOT / "dashboard" / "views.py").read_text(encoding="utf-8")
        production_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "dashboard").glob("*.py"))
        )

        fusion_start = views.index("def _render_prediction_fusion_overview")
        eeg_start = views.index("def _render_eeg_pretraining")
        fusion_source = views[fusion_start:eeg_start]

        self.assertNotIn("pretraining_kpis", fusion_source)
        self.assertNotIn("EEG 预训练产物", fusion_source)
        self.assertNotIn("EEG 预训练产物", production_source)

    def test_eeg_benchmark_selection_reasons_are_model_cards(self):
        views = (ROOT / "dashboard" / "views.py").read_text(encoding="utf-8")
        components = (ROOT / "dashboard" / "components.py").read_text(encoding="utf-8")

        eeg_start = views.index("def _render_eeg_pretraining")
        fmri_start = views.index("def _render_fmri_pretraining")
        eeg_source = views[eeg_start:fmri_start]

        self.assertIn("render_model_reason_grid", eeg_source)
        self.assertIn("render_ablation_card_grid", eeg_source)
        self.assertNotIn("overview[\"selection_standards\"]", eeg_source)
        self.assertNotIn("st.dataframe", eeg_source)
        self.assertNotIn("数据目录", eeg_source)
        self.assertNotIn("pretraining.root", eeg_source)
        self.assertIn("model-reason-grid", components)
        self.assertIn("model-reason-card", components)
        self.assertIn("ablation-card-grid", components)

    def test_eeg_assets_prefer_best_version_outputs(self):
        data = (ROOT / "dashboard" / "data.py").read_text(encoding="utf-8")

        self.assertIn("runs\" / \"_best\"", data)
        self.assertIn("roc_test.png", data)
        self.assertIn("confusion_test.png", data)
        self.assertNotIn("04_roc_source.png", data)

    def test_final_copy_has_no_prompt_phrasing(self):
        rendered_sources = [
            (ROOT / "dashboard" / "views.py").read_text(encoding="utf-8"),
            (ROOT / "dashboard" / "components.py").read_text(encoding="utf-8"),
        ]
        forbidden = [
            "医院方",
            "科研导师",
            "给医生",
            "给 AI",
            "Maternal AI Research",
            "AI Research",
            "提示词",
            "首页只",
            "汇报重点",
            "研究者方法",
            "保留给研究者",
        ]

        combined = "\n".join(rendered_sources)
        for phrase in forbidden:
            self.assertNotIn(phrase, combined)

    def test_hero_uses_simple_functional_illustration(self):
        components = (ROOT / "dashboard" / "components.py").read_text(encoding="utf-8")

        self.assertIn("pipeline-illustration", components)
        self.assertIn("source-node physiology", components)
        self.assertIn("source-node eeg", components)
        self.assertIn("source-node fmri", components)
        self.assertIn("fusion-core", components)
        self.assertIn("risk-output", components)
        self.assertIn("signal-wave", components)


if __name__ == "__main__":
    unittest.main()
