from pathlib import Path
import unittest

from dashboard.config import APP_ROOT, PRETRAIN_ROOT, RESULTS_DIR
from dashboard.data import load_physiology_summary, load_pretraining_data
from dashboard.services import (
    build_feature_cloud_terms,
    build_pretraining_overview,
    dashboard_python_files,
    method_outcomes,
)


class DashboardServiceTests(unittest.TestCase):
    def test_pretraining_root_prefers_external_eeg_nanda_directory(self):
        expected = Path("E:/STUDY_files/EEG_NANDA")

        if expected.exists():
            self.assertEqual(PRETRAIN_ROOT, expected)
            self.assertTrue((PRETRAIN_ROOT / "experiments").exists())
            self.assertTrue((PRETRAIN_ROOT / "results_plots_source").exists())

    def test_physiology_summary_adds_baseline_lifts(self):
        summary = load_physiology_summary(RESULTS_DIR)

        self.assertEqual(len(summary), 12)
        self.assertIn("pr_lift", summary.columns)
        self.assertIn("roc_lift", summary.columns)

        first = summary.iloc[0]
        expected_lift = first["pr_auc_median"] - first["baseline_pr_auc"]
        self.assertAlmostEqual(first["pr_lift"], expected_lift, places=12)

        pr_scm, pr_linear, pr_tie, pr_total = method_outcomes(summary, "pr_auc_median")
        self.assertEqual(pr_total, 6)
        self.assertGreaterEqual(pr_scm + pr_linear + pr_tie, pr_total)

    def test_feature_cloud_terms_are_ranked(self):
        summary = load_physiology_summary(RESULTS_DIR)

        terms = build_feature_cloud_terms(summary)

        self.assertGreaterEqual(len(terms), 4)
        self.assertEqual(terms.iloc[0]["weight"], terms["weight"].max())
        self.assertIn("BMI", set(terms["feature"]))

    def test_pretraining_overview_prioritizes_route_b(self):
        pretraining = load_pretraining_data(PRETRAIN_ROOT)

        overview = build_pretraining_overview(pretraining)

        primary_versions = {row["version"] for row in overview["primary_rows"]}
        self.assertIn("V2", primary_versions)
        self.assertNotIn("路线A", str(overview))
        self.assertNotIn("路线B", str(overview))
        self.assertGreaterEqual(len(overview["asset_images"]), 3)
        self.assertIn("heldout_auc", overview["risk"])

    def test_pretraining_overview_includes_benchmarks_and_modality_status(self):
        pretraining = load_pretraining_data(PRETRAIN_ROOT)

        overview = build_pretraining_overview(pretraining)

        self.assertEqual(overview["modalities"]["eeg"]["status"], "已完成")
        self.assertEqual(overview["modalities"]["fmri"]["status"], "待训练")
        self.assertGreaterEqual(len(overview["benchmarks"]), 5)
        self.assertGreaterEqual(len(overview["selection_standards"]), 4)
        reasons = " ".join(str(row.get("reason", "")) for row in overview["benchmarks"])
        for row in overview["benchmarks"]:
            self.assertTrue(str(row.get("reason", "")).strip(), row["model"])
        self.assertIn("开源", reasons)
        self.assertIn("原始数据集", reasons)
        self.assertIn("SOTA", reasons)
        best = overview["best_version"]
        self.assertEqual(best["version"], "V2")
        self.assertAlmostEqual(best["test_auc"], 0.75, places=3)

    def test_dashboard_files_stay_under_500_lines(self):
        files = dashboard_python_files(APP_ROOT)

        self.assertIn(APP_ROOT / "app.py", files)
        for path in files:
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(line_count, 500, f"{path} has {line_count} lines")


if __name__ == "__main__":
    unittest.main()
