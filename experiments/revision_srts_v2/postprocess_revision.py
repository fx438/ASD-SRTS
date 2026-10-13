#!/usr/bin/env python3
"""对标准五折修订实验做完整性核验、配对 bootstrap 与制表。

本脚本只读取已经完成的结果，不训练模型，也不覆盖任何已有文件。
主结果仍报告五折均值±sample SD；bootstrap CI 针对固定五个折模型产生的
860 行 OOF 预测，是条件于本次训练结果的预测层不确定性，不代表多随机种子波动。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

CORE_VARIANTS = (
    "raw",
    "shrink_fisher",
    "eigclip_tangent",
    "lw_tangent",
    "srts",
)
METRICS = ("auc", "accuracy", "sensitivity", "specificity", "precision", "f1")
DISPLAY_NAMES = {
    "raw": "原始 Pearson FC",
    "shrink_fisher": "线性收缩+Fisher-Z",
    "eigclip_tangent": "特征值截断+切线映射",
    "lw_tangent": "Ledoit-Wolf+切线映射",
    "srts": "完整 SRTS",
    "raw_mixup": "原始 FC+潜表征 Mixup",
    "srts_mixup": "SRTS+潜表征 Mixup",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_text_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: float, b: float, atol: float = 1e-9) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=atol))


def hard_metrics(labels: np.ndarray, probabilities: np.ndarray,
                 predictions: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.int64, copy=False)
    predictions = predictions.astype(np.int64, copy=False)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (2.0 * precision * sensitivity / (precision + sensitivity)
          if precision + sensitivity else 0.0)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(np.mean(predictions == labels)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
    }


def load_inputs(result_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]],
                                            dict[str, Any], dict[str, Any] | None]:
    summary = read_json(result_dir / "summary.json")
    folds = [read_json(result_dir / "folds" / f"fold_{index}.json")
             for index in range(1, 6)]
    splits = read_json(result_dir / "splits_manifest.json")
    mixup_path = result_dir / "mixup_test_results.json"
    mixup_test = read_json(mixup_path) if mixup_path.exists() else None
    return summary, folds, splits, mixup_test


def validate_integrity(result_dir: Path, summary: dict[str, Any],
                       folds: list[dict[str, Any]], splits_doc: dict[str, Any],
                       mixup_test: dict[str, Any] | None) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    protocol = summary["protocol"]
    checks["protocol_sizes_are_602_86_172"] = (
        protocol["per_fold_sizes"] == [602, 86, 172]
        and all(fold["sizes"] == [602, 86, 172] for fold in folds)
    )
    checks["sample_sd_declared_ddof_1"] = "ddof=1" in protocol["sd_definition"]
    checks["fold_ids_are_0_to_4"] = [int(fold["fold"]) for fold in folds] == list(range(5))

    if isinstance(splits_doc, dict):
        split_rows = splits_doc.get("splits", splits_doc.get("folds", splits_doc))
    else:
        split_rows = splits_doc
    if not isinstance(split_rows, list) or len(split_rows) != 5:
        raise AssertionError("splits_manifest.json 不是 5 折列表")
    test_count = np.zeros(860, dtype=np.int64)
    split_disjoint = True
    split_union = True
    fold_matches = True
    for fold, split in zip(folds, split_rows):
        train = np.asarray(split["train"], dtype=np.int64)
        val = np.asarray(split["val"], dtype=np.int64)
        test = np.asarray(split["test"], dtype=np.int64)
        split_disjoint &= (
            len(np.intersect1d(train, val)) == 0
            and len(np.intersect1d(train, test)) == 0
            and len(np.intersect1d(val, test)) == 0
        )
        split_union &= (
            len(train) == 602 and len(val) == 86 and len(test) == 172
            and np.array_equal(np.sort(np.concatenate([train, val, test])), np.arange(860))
        )
        fold_matches &= np.array_equal(test, np.asarray(fold["test_indices"], dtype=np.int64))
        test_count[test] += 1
    checks["within_fold_splits_are_disjoint"] = bool(split_disjoint)
    checks["each_fold_partitions_all_860_subjects"] = bool(split_union)
    checks["fold_json_test_indices_match_locked_splits"] = bool(fold_matches)
    checks["each_subject_is_tested_exactly_once"] = bool(np.all(test_count == 1))
    details["test_count_distribution"] = {
        str(value): int(np.sum(test_count == value)) for value in np.unique(test_count)
    }

    oof = summary["oof_predictions"]
    indices = np.asarray([row["subject_index"] for row in oof], dtype=np.int64)
    labels = np.asarray([row["label"] for row in oof], dtype=np.int64)
    checks["oof_has_exactly_860_rows"] = len(oof) == 860
    checks["oof_subject_indices_are_unique_0_to_859"] = np.array_equal(indices, np.arange(860))
    checks["oof_class_counts_are_417_443"] = (
        int(np.sum(labels == 1)) == 417 and int(np.sum(labels == 0)) == 443
    )

    probability_ok = True
    label_match = True
    metric_match = True
    selected_gamma_match = True
    diagnostics_match = True
    for fold in folds:
        fold_id = int(fold["fold"])
        test_indices = np.asarray(fold["test_indices"], dtype=np.int64)
        trials = fold["gamma_trials"]
        expected_trial = max(
            enumerate(trials), key=lambda item: (float(item[1]["best_val_auc"]), -item[0])
        )[1]
        selected_gamma_match &= close(expected_trial["gamma"], fold["selected_gamma"], 1e-12)
        diagnostics_match &= (
            len(trials) == 8
            and all(len(trial["numerics"]) == 86 for trial in trials)
            and all("test" not in trial for trial in trials)
        )
        for variant in CORE_VARIANTS:
            test = fold["variants"][variant]["test"]
            probabilities = np.asarray(test["probs"], dtype=float)
            test_labels = np.asarray(test["labels"], dtype=np.int64)
            threshold = float(test["threshold"])
            probability_ok &= (
                len(probabilities) == 172
                and np.all(np.isfinite(probabilities))
                and np.all((probabilities >= 0.0) & (probabilities <= 1.0))
            )
            label_match &= np.array_equal(test_labels, labels[test_indices])
            recalculated = hard_metrics(
                test_labels, probabilities, (probabilities >= threshold).astype(np.int64)
            )
            metric_match &= all(close(recalculated[name], test[name], 1e-9)
                                for name in METRICS)
            oof_probs = np.asarray(
                [oof[index][f"{variant}_prob"] for index in test_indices], dtype=float
            )
            metric_match &= np.allclose(oof_probs, probabilities, rtol=0.0, atol=1e-12)
            summary_fold_values = summary["variants"][variant]["auc"]["fold_values"]
            metric_match &= close(summary_fold_values[fold_id], test["auc"], 1e-12)
    checks["selected_gamma_is_validation_argmax_with_small_gamma_tie_break"] = bool(selected_gamma_match)
    checks["gamma_trials_have_8_candidates_86_diagnostics_and_no_test_field"] = bool(diagnostics_match)
    checks["all_probabilities_are_finite_and_in_unit_interval"] = bool(probability_ok)
    checks["fold_test_labels_match_oof_manifest_labels"] = bool(label_match)
    checks["saved_test_metrics_and_oof_probabilities_recompute_exactly"] = bool(metric_match)

    decision = summary["mixup_decision"]
    deltas = np.asarray(
        [fold["mixup_validation_only"]["srts_mixup"]["best_val_auc"]
         - fold["variants"]["srts"]["best_val_auc"] for fold in folds], dtype=float
    )
    expected_accept = bool(
        float(np.mean(deltas)) >= float(decision["rule"]["minimum_mean_validation_auc_delta"])
        and int(np.sum(deltas > 0.0)) >= int(decision["rule"]["minimum_improved_folds"])
    )
    checks["mixup_gate_recomputes_from_validation_only"] = (
        np.allclose(deltas, decision["validation_auc_deltas_srts_mixup_minus_srts"],
                    rtol=0.0, atol=1e-12)
        and close(np.mean(deltas), decision["mean_validation_auc_delta"], 1e-12)
        and int(np.sum(deltas > 0.0)) == int(decision["improved_folds"])
        and expected_accept == bool(decision["accepted"])
        and decision["selection_uses_test_data"] is False
    )
    checks["mixup_test_file_presence_matches_gate"] = (
        (mixup_test is not None) == bool(decision["accepted"])
        and bool(decision["test_evaluated"]) == bool(decision["accepted"])
    )
    details["mixup_validation_deltas"] = deltas.tolist()

    provenance_path = result_dir / "provenance.json"
    config_path = result_dir / "config.json"
    details["input_sha256"] = {
        "summary.json": sha256(result_dir / "summary.json"),
        "splits_manifest.json": sha256(result_dir / "splits_manifest.json"),
        "provenance.json": sha256(provenance_path),
        "config.json": sha256(config_path),
        **{f"folds/fold_{i}.json": sha256(result_dir / "folds" / f"fold_{i}.json")
           for i in range(1, 6)},
    }
    checks["all_integrity_checks_pass"] = all(checks.values())
    report = {"checks": checks, "details": details}
    if not checks["all_integrity_checks_pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError("完整性核验失败：" + ", ".join(failed))
    return report


def build_oof_arrays(summary: dict[str, Any], folds: list[dict[str, Any]],
                     mixup_test: dict[str, Any] | None
                     ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray],
                                dict[str, np.ndarray]]:
    oof = summary["oof_predictions"]
    labels = np.asarray([row["label"] for row in oof], dtype=np.int64)
    fold_ids = np.asarray([row["fold"] for row in oof], dtype=np.int64)
    variants = list(CORE_VARIANTS)
    if mixup_test is not None:
        variants.extend(("raw_mixup", "srts_mixup"))
    probabilities = {
        variant: np.asarray([row[f"{variant}_prob"] for row in oof], dtype=float)
        for variant in variants
    }
    predictions: dict[str, np.ndarray] = {
        variant: np.zeros(860, dtype=np.int64) for variant in variants
    }
    for fold in folds:
        test_indices = np.asarray(fold["test_indices"], dtype=np.int64)
        for variant in CORE_VARIANTS:
            threshold = float(fold["variants"][variant]["test"]["threshold"])
            predictions[variant][test_indices] = (
                probabilities[variant][test_indices] >= threshold
            ).astype(np.int64)
    if mixup_test is not None:
        for variant in ("raw_mixup", "srts_mixup"):
            for record in mixup_test[variant]:
                indices = np.asarray(record["test_indices"], dtype=np.int64)
                threshold = float(record["test"]["threshold"])
                predictions[variant][indices] = (
                    probabilities[variant][indices] >= threshold
                ).astype(np.int64)
    return labels, fold_ids, probabilities, predictions


def paired_bootstrap(labels: np.ndarray, fold_ids: np.ndarray,
                     probabilities: dict[str, np.ndarray],
                     predictions: dict[str, np.ndarray], comparisons: list[tuple[str, str]],
                     n_bootstrap: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero((fold_ids == fold) & (labels == label))
              for fold in range(5) for label in (0, 1)]
    point_metrics = {
        variant: hard_metrics(labels, probabilities[variant], predictions[variant])
        for variant in probabilities
    }
    draws: dict[tuple[str, str], dict[str, list[float]]] = {
        comparison: {metric: [] for metric in METRICS} for comparison in comparisons
    }
    for _ in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata
        ])
        sampled_labels = labels[sampled]
        sample_metrics = {
            variant: hard_metrics(sampled_labels, probability[sampled],
                                  predictions[variant][sampled])
            for variant, probability in probabilities.items()
        }
        for numerator, denominator in comparisons:
            for metric in METRICS:
                draws[(numerator, denominator)][metric].append(
                    sample_metrics[numerator][metric] - sample_metrics[denominator][metric]
                )

    results: list[dict[str, Any]] = []
    for numerator, denominator in comparisons:
        record: dict[str, Any] = {
            "comparison": f"{numerator}_minus_{denominator}",
            "numerator": numerator,
            "denominator": denominator,
            "metrics": {},
        }
        for metric in METRICS:
            values = np.asarray(draws[(numerator, denominator)][metric], dtype=float)
            record["metrics"][metric] = {
                "point_difference": float(
                    point_metrics[numerator][metric] - point_metrics[denominator][metric]
                ),
                "percentile_95_ci": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
                "bootstrap_mean_difference": float(values.mean()),
                "bootstrap_sample_sd": float(values.std(ddof=1)),
            }
        results.append(record)
    return {
        "method": "paired stratified within-fold subject bootstrap",
        "estimand": "pooled OOF metric difference conditional on the five fitted fold models",
        "limitations": (
            "CI 不包含重新划分、重新训练或随机初始化带来的不确定性，"
            "不能解释为多随机种子置信区间。"
        ),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "point_oof_metrics": point_metrics,
        "comparisons": results,
    }


def summarize_gamma(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_gamma: dict[float, dict[str, list[float]]] = {}
    for fold in folds:
        for trial in fold["gamma_trials"]:
            gamma = float(trial["gamma"])
            item = by_gamma.setdefault(gamma, {
                "val_auc": [], "min_eigenvalue": [], "log10_condition_number": []
            })
            item["val_auc"].append(float(trial["best_val_auc"]))
            item["min_eigenvalue"].extend(
                float(row["min_eigenvalue"]) for row in trial["numerics"]
            )
            item["log10_condition_number"].extend(
                float(row["log10_condition_number"]) for row in trial["numerics"]
            )
    records = []
    for gamma, values in sorted(by_gamma.items()):
        val = np.asarray(values["val_auc"], dtype=float)
        eigen = np.asarray(values["min_eigenvalue"], dtype=float)
        condition = np.asarray(values["log10_condition_number"], dtype=float)
        records.append({
            "gamma": gamma,
            "validation_auc_mean": float(val.mean()),
            "validation_auc_sample_sd": float(val.std(ddof=1)),
            "validation_auc_fold_values": val.tolist(),
            "minimum_eigenvalue_global_min": float(eigen.min()),
            "minimum_eigenvalue_q25": float(np.quantile(eigen, 0.25)),
            "minimum_eigenvalue_median": float(np.median(eigen)),
            "minimum_eigenvalue_q75": float(np.quantile(eigen, 0.75)),
            "log10_condition_number_min": float(condition.min()),
            "log10_condition_number_q25": float(np.quantile(condition, 0.25)),
            "log10_condition_number_median": float(np.median(condition)),
            "log10_condition_number_q75": float(np.quantile(condition, 0.75)),
            "log10_condition_number_max": float(condition.max()),
            "n_fold_auc": int(len(val)),
            "n_validation_diagnostics": int(len(eigen)),
        })
    return records


def write_csv_new(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_tables(out_dir: Path, summary: dict[str, Any], bootstrap: dict[str, Any],
                gamma: list[dict[str, Any]]) -> None:
    metric_rows = []
    for variant, values in summary["variants"].items():
        row: dict[str, Any] = {
            "variant": variant,
            "display_name": DISPLAY_NAMES[variant],
            "mean_validation_auc": values["mean_validation_auc"],
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = values[metric]["mean"]
            row[f"{metric}_sample_sd"] = values[metric]["sample_sd"]
            for fold, value in enumerate(values[metric]["fold_values"], start=1):
                row[f"{metric}_fold_{fold}"] = value
        metric_rows.append(row)
    metric_fields = ["variant", "display_name", "mean_validation_auc"] + [
        field for metric in METRICS
        for field in ([f"{metric}_mean", f"{metric}_sample_sd"]
                      + [f"{metric}_fold_{fold}" for fold in range(1, 6)])
    ]
    write_csv_new(out_dir / "metrics_summary.csv", metric_rows, metric_fields)

    gamma_fields = [
        "gamma", "validation_auc_mean", "validation_auc_sample_sd",
        "validation_auc_fold_1", "validation_auc_fold_2", "validation_auc_fold_3",
        "validation_auc_fold_4", "validation_auc_fold_5",
        "minimum_eigenvalue_global_min", "minimum_eigenvalue_q25",
        "minimum_eigenvalue_median", "minimum_eigenvalue_q75",
        "log10_condition_number_min", "log10_condition_number_q25",
        "log10_condition_number_median", "log10_condition_number_q75",
        "log10_condition_number_max", "n_fold_auc", "n_validation_diagnostics",
    ]
    gamma_rows = []
    for record in gamma:
        row = dict(record)
        for index, value in enumerate(record["validation_auc_fold_values"], start=1):
            row[f"validation_auc_fold_{index}"] = value
        gamma_rows.append(row)
    write_csv_new(out_dir / "gamma_diagnostics.csv", gamma_rows, gamma_fields)

    difference_rows = []
    for comparison in bootstrap["comparisons"]:
        row = {
            "comparison": comparison["comparison"],
            "numerator": comparison["numerator"],
            "denominator": comparison["denominator"],
        }
        for metric, values in comparison["metrics"].items():
            row[f"{metric}_difference"] = values["point_difference"]
            row[f"{metric}_ci_low"] = values["percentile_95_ci"][0]
            row[f"{metric}_ci_high"] = values["percentile_95_ci"][1]
        difference_rows.append(row)
    difference_fields = ["comparison", "numerator", "denominator"] + [
        field for metric in METRICS
        for field in (f"{metric}_difference", f"{metric}_ci_low", f"{metric}_ci_high")
    ]
    write_csv_new(out_dir / "paired_oof_differences.csv", difference_rows, difference_fields)


def make_oof_csv(out_dir: Path, summary: dict[str, Any],
                 predictions: dict[str, np.ndarray]) -> None:
    rows = []
    for index, source in enumerate(summary["oof_predictions"]):
        row = dict(source)
        for variant, values in predictions.items():
            row[f"{variant}_prediction"] = int(values[index])
        rows.append(row)
    fieldnames = list(rows[0].keys())
    write_csv_new(out_dir / "oof_predictions.csv", rows, fieldnames)


def format_metric(value: dict[str, Any]) -> str:
    return f'{value["mean"]:.4f}±{value["sample_sd"]:.4f}'


def make_markdown(summary: dict[str, Any], bootstrap: dict[str, Any],
                  gamma: list[dict[str, Any]]) -> str:
    lines = [
        "# 标准五折修订实验结果（自动生成）",
        "",
        "> 本文件由锁定的 `summary.json`、五个折结果和 OOF 预测自动生成。",
        "> 主指标为五折均值±sample SD（`ddof=1`）；配对 bootstrap CI 条件于本次固定的五个折模型。",
        "",
        "## 完整性结论",
        "",
        "- 每折 602/86/172 例；五个测试集并集为 860 例；每名被试恰好测试一次。",
        "- γ 逐折仅按该折验证 AUC 选择；测试结果未出现在 γ 候选记录中。",
        f'- 各折选择的 γ：{summary["selected_gammas"]}。',
        f'- Mixup 验证门：`accepted={summary["mixup_decision"]["accepted"]}`，'
        f'平均验证 AUC 差值 {summary["mixup_decision"]["mean_validation_auc_delta"]:+.4f}，'
        f'提升折数 {summary["mixup_decision"]["improved_folds"]}/5。',
        "",
        "## 五折测试结果",
        "",
        "| 表示/模型 | AUC | ACC | SEN | SPE | F1 | 平均验证 AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, values in summary["variants"].items():
        lines.append(
            f'| {DISPLAY_NAMES[variant]} | {format_metric(values["auc"])} | '
            f'{format_metric(values["accuracy"])} | {format_metric(values["sensitivity"])} | '
            f'{format_metric(values["specificity"])} | {format_metric(values["f1"])} | '
            f'{values["mean_validation_auc"]:.4f} |'
        )
    lines.extend([
        "",
        "## SRTS 的配对 OOF 差值",
        "",
        "| 比较 | ΔAUC（95% CI） | ΔACC（95% CI） |",
        "|---|---:|---:|",
    ])
    for comparison in bootstrap["comparisons"]:
        auc = comparison["metrics"]["auc"]
        acc = comparison["metrics"]["accuracy"]
        name = f'{DISPLAY_NAMES[comparison["numerator"]]} − {DISPLAY_NAMES[comparison["denominator"]]}'
        lines.append(
            f'| {name} | {auc["point_difference"]:+.4f} '
            f'[{auc["percentile_95_ci"][0]:+.4f}, {auc["percentile_95_ci"][1]:+.4f}] | '
            f'{acc["point_difference"]:+.4f} '
            f'[{acc["percentile_95_ci"][0]:+.4f}, {acc["percentile_95_ci"][1]:+.4f}] |'
        )
    lines.extend([
        "",
        "说明：上述 CI 使用折内、类别分层的被试级配对 bootstrap；不包含重新训练、重新划分或随机初始化的不确定性。",
        "",
        "## γ 验证与数值诊断",
        "",
        "| γ | 验证 AUC（均值±SD） | 白化矩阵最小特征值中位数 [Q1,Q3] | log10(条件数)中位数 [Q1,Q3] |",
        "|---:|---:|---:|---:|",
    ])
    for record in gamma:
        lines.append(
            f'| {record["gamma"]:.2f} | {record["validation_auc_mean"]:.4f}±'
            f'{record["validation_auc_sample_sd"]:.4f} | '
            f'{record["minimum_eigenvalue_median"]:.3e} '
            f'[{record["minimum_eigenvalue_q25"]:.3e}, {record["minimum_eigenvalue_q75"]:.3e}] | '
            f'{record["log10_condition_number_median"]:.3f} '
            f'[{record["log10_condition_number_q25"]:.3f}, '
            f'{record["log10_condition_number_q75"]:.3f}] |'
        )
    lw = summary["ledoit_wolf_shrinkage"]
    lines.extend([
        "",
        "## Ledoit-Wolf 收缩系数",
        "",
        f'- n={lw["n"]}，均值={lw["mean"]:.4f}，sample SD={lw["sample_sd"]:.4f}。',
        f'- 中位数={lw["median"]:.4f}，四分位区间=[{lw["q25"]:.4f}, {lw["q75"]:.4f}]，'
        f'范围=[{lw["min"]:.4f}, {lw["max"]:.4f}]。',
        "",
    ])
    return "\n".join(lines)


def make_gamma_figure(out_dir: Path, gamma_records: list[dict[str, Any]],
                      selected_gammas: list[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gamma = np.asarray([row["gamma"] for row in gamma_records], dtype=float)
    auc = np.asarray([row["validation_auc_mean"] for row in gamma_records], dtype=float)
    auc_sd = np.asarray([row["validation_auc_sample_sd"] for row in gamma_records], dtype=float)
    eig_med = np.asarray([row["minimum_eigenvalue_median"] for row in gamma_records])
    eig_q1 = np.asarray([row["minimum_eigenvalue_q25"] for row in gamma_records])
    eig_q3 = np.asarray([row["minimum_eigenvalue_q75"] for row in gamma_records])
    cond_med = np.asarray([row["log10_condition_number_median"] for row in gamma_records])
    cond_q1 = np.asarray([row["log10_condition_number_q25"] for row in gamma_records])
    cond_q3 = np.asarray([row["log10_condition_number_q75"] for row in gamma_records])

    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    axes[0].errorbar(gamma, auc, yerr=auc_sd, marker="o", capsize=3,
                     color="#315b7d", linewidth=1.6)
    axes[0].set_ylabel("Validation AUC")
    axes[0].set_title("(a) Validation performance")
    axes[0].grid(alpha=0.22)

    axes[1].plot(gamma, eig_med, marker="o", color="#8f4e3f", linewidth=1.6)
    axes[1].fill_between(gamma, eig_q1, eig_q3, color="#8f4e3f", alpha=0.18)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Minimum eigenvalue")
    axes[1].set_title("(b) Numerical positivity")
    axes[1].grid(alpha=0.22)

    axes[2].plot(gamma, cond_med, marker="o", color="#4e7652", linewidth=1.6)
    axes[2].fill_between(gamma, cond_q1, cond_q3, color="#4e7652", alpha=0.18)
    axes[2].set_ylabel(r"$\log_{10}$(condition number)")
    axes[2].set_title("(c) Numerical conditioning")
    axes[2].grid(alpha=0.22)
    for axis in axes:
        axis.set_xlabel(r"Shrinkage coefficient $\gamma$")
        axis.set_xticks(gamma)
        axis.tick_params(axis="x", rotation=45)
    selection_text = ", ".join(f"{value:.2f}" for value in selected_gammas)
    fig.suptitle(f"Fold-wise selected γ: {selection_text}", fontsize=10.5)
    fig.savefig(out_dir / "gamma_validation_numerics.png", dpi=300,
                facecolor="white", bbox_inches="tight")
    fig.savefig(out_dir / "gamma_validation_numerics.svg", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    out_dir = result_dir / "postprocess"
    out_dir.mkdir(parents=False, exist_ok=False)
    summary, folds, splits, mixup_test = load_inputs(result_dir)
    integrity = validate_integrity(result_dir, summary, folds, splits, mixup_test)
    labels, fold_ids, probabilities, predictions = build_oof_arrays(
        summary, folds, mixup_test
    )
    comparisons = [("srts", comparator) for comparator in (
        "raw", "shrink_fisher", "eigclip_tangent", "lw_tangent"
    )]
    if mixup_test is not None:
        comparisons.append(("srts_mixup", "srts"))
    bootstrap = paired_bootstrap(
        labels, fold_ids, probabilities, predictions, comparisons,
        n_bootstrap=args.bootstrap, seed=args.seed,
    )
    gamma = summarize_gamma(folds)

    write_json_new(out_dir / "integrity_report.json", integrity)
    write_json_new(out_dir / "paired_bootstrap.json", bootstrap)
    write_json_new(out_dir / "gamma_diagnostics.json", gamma)
    make_tables(out_dir, summary, bootstrap, gamma)
    make_oof_csv(out_dir, summary, predictions)
    write_text_new(out_dir / "manuscript_results.md",
                   make_markdown(summary, bootstrap, gamma))
    make_gamma_figure(out_dir, gamma, summary["selected_gammas"])
    provenance = {
        "script_sha256": sha256(Path(__file__).resolve()),
        "result_dir": str(result_dir),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "input_sha256": integrity["details"]["input_sha256"],
    }
    write_json_new(out_dir / "postprocess_provenance.json", provenance)
    print(f"完整性核验通过；后处理结果已写入：{out_dir}")


if __name__ == "__main__":
    main()
