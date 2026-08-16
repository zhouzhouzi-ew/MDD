from pathlib import Path
import hashlib
import unittest

from dashboard.config import APP_ROOT, PRETRAIN_ROOT
from dashboard.data import load_pretraining_data
from dashboard.services import build_pretraining_overview


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PredictionModelAssetTests(unittest.TestCase):
    def test_prediction_model_copy_uses_readable_chinese(self):
        sources = {
            "views.py": (APP_ROOT / "dashboard" / "views.py").read_text(encoding="utf-8"),
            "components.py": (APP_ROOT / "dashboard" / "components.py").read_text(encoding="utf-8"),
        }

        self.assertIn('"eeg": {"label": "预测模型"', sources["components.py"])
        for text in (
            "预测模型",
            "融合总览",
            "EEG 预训练",
            "fMRI 预训练",
            "现有模型对照",
            "版本升级路线图",
            "待训练",
        ):
            self.assertIn(text, sources["views.py"] + sources["components.py"])

    def test_prediction_model_uses_clean_static_eeg_figures(self):
        pretraining = load_pretraining_data(PRETRAIN_ROOT)
        overview = build_pretraining_overview(pretraining)

        expected = [
            ("v2_connectivity_7net.png", "7网络连接差异"),
            ("v2_node_strength_top24.png", "节点强度 Top24"),
            ("roc_test.png", "测试集 ROC"),
            ("confusion_test.png", "测试集混淆矩阵"),
            ("training_curves.png", "训练曲线"),
        ]
        static_dir = APP_ROOT / "dashboard" / "static" / "eeg"

        self.assertEqual(
            [(Path(item["path"]).name, item["title"]) for item in overview["asset_images"]],
            expected,
        )
        self.assertTrue(all(Path(item["path"]).parent == static_dir for item in overview["asset_images"]))

    def test_static_eeg_figures_match_reference_outputs(self):
        source_pairs = [
            (
                APP_ROOT / "dashboard" / "static" / "eeg" / "v2_connectivity_7net.png",
                PRETRAIN_ROOT / "results_plots_source" / "02_connectivity_7net.png",
            ),
            (
                APP_ROOT / "dashboard" / "static" / "eeg" / "v2_node_strength_top24.png",
                PRETRAIN_ROOT / "results_plots_source" / "03_node_strength_top24.png",
            ),
            (
                APP_ROOT / "dashboard" / "static" / "eeg" / "roc_test.png",
                PRETRAIN_ROOT / "experiments" / "02_multiband_gnn" / "runs" / "_best" / "roc_test.png",
            ),
            (
                APP_ROOT / "dashboard" / "static" / "eeg" / "confusion_test.png",
                PRETRAIN_ROOT / "experiments" / "02_multiband_gnn" / "runs" / "_best" / "confusion_test.png",
            ),
            (
                APP_ROOT / "dashboard" / "static" / "eeg" / "training_curves.png",
                PRETRAIN_ROOT / "experiments" / "02_multiband_gnn" / "runs" / "_best" / "training_curves.png",
            ),
        ]

        for static_path, reference_path in source_pairs:
            self.assertTrue(static_path.exists(), static_path)
            self.assertTrue(reference_path.exists(), reference_path)
            self.assertEqual(_sha256(static_path), _sha256(reference_path), static_path.name)

    def test_mumtaz_benchmarks_are_disambiguated(self):
        pretraining = load_pretraining_data(PRETRAIN_ROOT)
        overview = build_pretraining_overview(pretraining)

        mumtaz_rows = [row for row in overview["benchmarks"] if "Mumtaz" in row["model"]]

        self.assertEqual(
            [row["model"] for row in mumtaz_rows],
            [
                "Mumtaz 2017 LR (spectral features)",
                "Mumtaz 2018 LR (connectivity features)",
            ],
        )
        self.assertTrue(all(row["reason"].strip() for row in mumtaz_rows))


if __name__ == "__main__":
    unittest.main()
