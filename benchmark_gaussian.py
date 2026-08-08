"""Gaussian benchmark: PR-AUC / ROC-AUC that a classifier hitting
precision=0.6 AND recall=0.6 (equal-variance Gaussian scores) corresponds to.

Model: positives ~ N(d, 1), negatives ~ N(0, 1); predict positive if score > t.
Given the class prior pi and a target operating point (p, r), solve for d, then
report the full-curve ROC-AUC and PR-AUC (average precision) of that model.
"""
import numpy as np
from scipy.stats import norm

def phi(x):
    return norm.cdf(x)

def solve_separation(pi, p_target, r_target):
    """Equal-variance Gaussian separation d and threshold t hitting (p, r)."""
    fpr = pi * r_target * (1 - p_target) / (p_target * (1 - pi))  # FP rate required
    t = norm.ppf(1 - fpr)
    d = t - norm.ppf(1 - r_target)  # recall = 1 - Phi(t - d)
    return d, t, fpr

def pr_auc_from_d(d, pi, n_grid=400_001):
    """Average precision by deterministic threshold sweep."""
    t_vals = np.linspace(-8.0, d + 8.0, n_grid)
    tpr = 1.0 - phi(t_vals - d)
    fpr = 1.0 - phi(t_vals)
    num = pi * tpr
    den = num + (1.0 - pi) * fpr
    precision = np.divide(num, den, out=np.ones_like(num), where=den > 0)
    # trapezoid over recall axis, descending threshold -> recall ascending
    recall = tpr[::-1]
    prec = precision[::-1]
    auc = np.trapezoid(prec, recall) if hasattr(np, "trapezoid") else np.trapz(prec, recall)
    return auc

def roc_auc_from_d(d):
    return phi(d / np.sqrt(2))

def precision_at_recall(d, pi, r):
    """Precision this Gaussian model reaches at the given recall."""
    t = d - norm.ppf(r)  # recall = 1 - Phi(t - d) => t = d - Phi^{-1}(r)
    fpr = 1.0 - phi(t)
    return pi * r / (pi * r + (1 - pi) * fpr)

def clinical_qualification():
    """Clinical-screening method: derive the 'qualified' ROC/PR-AUC from a
    tolerable false-positive rate + a minimum recall, not from an arbitrary AUC."""
    print("\n" + "=" * 72)
    print("临床筛查标准法:由'可承受 FPR' + '最低召回率' 推导合格线")
    print("=" * 72)
    defs = {"A": (12, 18190), "B": (93, 18271)}
    recall_target = 0.70  # 产后抑郁筛查通常要求召回率 60~80%,默认 70%
    print(f"设定:最低召回率 = {recall_target:.0%}")
    for label, (n_pos, n_total) in defs.items():
        pi = n_pos / n_total
        print(f"\n[定义{label}] 阳性率 = {pi:.5%} ({n_pos}/{n_total})")
        for fpr in (0.005, 0.01, 0.05):
            d = norm.ppf(1 - fpr) - norm.ppf(1 - recall_target)
            roc = roc_auc_from_d(d)
            prauc = pr_auc_from_d(d, pi)
            p_op = pi * recall_target / (pi * recall_target + (1 - pi) * fpr)
            fp_per_tp = (1 - pi) * fpr / (pi * recall_target)
            nnt = 1 / (pi * recall_target)
            print(
                f"  FPR={fpr:.1%}: 达标ROC-AUC={roc:.4f}  达标PR-AUC={prauc:.4f}  "
                f"工作点Precision={p_op:.2%}  | 每查1例真阳需随访{fp_per_tp:.1f}个假阳, 筛{nnt:,.0f}人"
            )

    print("\n当前模型在相同 FPR 下实际能达成的召回率(按高斯折算):")
    current = [("A", 0.726, 0.0035), ("B", 0.557, 0.0095)]
    for label, roc, prauc in current:
        d_cur = np.sqrt(2) * norm.ppf(roc)
        rows = []
        for fpr in (0.005, 0.01, 0.05):
            t = norm.ppf(1 - fpr)
            rows.append(f"FPR={fpr:.1%}->Recall≈{1 - phi(t - d_cur):.1%}")
        print(f"  定义{label} (ROC-AUC={roc:.3f}, PR-AUC={prauc:.4f}): " + ", ".join(rows))


def main():
    definitions = {
        "A (严格口径)": dict(n_pos=12, n_total=18190),
        "B (宽泛口径)": dict(n_pos=93, n_total=18271),
    }
    p_target, r_target = 0.6, 0.6

    print("=" * 72)
    print("高斯双类基准:在 PR=%.1f / Recall=%.1f 工作点上,该模型对应的整条曲线指标" % (p_target, r_target))
    print("=" * 72)
    for name, dct in definitions.items():
        pi = dct["n_pos"] / dct["n_total"]
        d, t, fpr = solve_separation(pi, p_target, r_target)
        roc = roc_auc_from_d(d)
        prauc = pr_auc_from_d(d, pi)
        # verify operating point
        p_check = precision_at_recall(d, pi, r_target)
        print(f"\n[{name}] 阳性率 = {pi:.5%} ({dct['n_pos']}/{dct['n_total']})")
        print(f"  所需高斯分离度 d            = {d:.3f}")
        print(f"  对应 ROC-AUC                = {roc:.4f}")
        print(f"  对应 PR-AUC (平均精度)      = {prauc:.4f}")
        print(f"  校验:该模型在 Recall=0.6 处 Precision = {p_check:.4f} (应为 0.6)")

    print("\n" + "=" * 72)
    print("当前模型实测 (results/scm_kernel_vs_linear_summary.csv 最优组合)")
    print("=" * 72)
    current = [
        ("A", 0.00352, 0.726, 12, 18190),
        ("B", 0.00950, 0.557, 93, 18271),
    ]
    for label, pr_med, roc_med, n_pos, n_total in current:
        pi = n_pos / n_total
        d_cur = np.sqrt(2) * norm.ppf(roc_med)
        p_at_06 = precision_at_recall(d_cur, pi, 0.6)
        print(f"\n定义{label}: PR-AUC={pr_med:.4f}, ROC-AUC={roc_med:.4f}")
        print(f"  按高斯假设折算,该模型在 Recall=0.6 处 Precision ≈ {p_at_06:.4%}")
        print(f"  与 60% 目标相比: 仅达到目标的 {p_at_06/0.6:.2%}")

    clinical_qualification()

if __name__ == "__main__":
    main()
