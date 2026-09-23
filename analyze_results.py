"""
Loads task{1,2,3,4}_summary.json (as produced by run_task1.py / evaluate_final.py /
evaluate_sketch.py / evaluate_osr.py) and produces the tables and figures you'd
actually want for the PA1 report. Missing files are skipped gracefully, so you
can run this now with just Tasks 1-3 and rerun once Task 4 finishes.

NONE of the saved figures carry an in-image title -- the filename itself IS
the (slugified) title, so you can tell what each figure is from its name in
your file browser. Legends are still drawn where there's more than one
series to distinguish.

Usage:
    python analyze_results.py --results_dir /path/to/folder/with/jsons --out_dir ./figures

Expects (any subset present):
    task1_summary.json, task2_summary.json, task3_summary.json, task4_summary.json
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def slugify(title: str) -> str:
    """'Task 1: Clean baseline accuracy' -> 'task1_clean_baseline_accuracy'"""
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def savefig(fig, out_dir, title):
    """Saves fig as <slug(title)>.png -- NO title is drawn on the figure
    itself; the descriptive filename carries that information instead."""
    os.makedirs(out_dir, exist_ok=True)
    fname = slugify(title) + ".png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# =============================================================================
# Task 1
# =============================================================================

def task1_tables_and_plots(d, out_dir):
    if d is None:
        print("[task1] no summary found, skipping")
        return
    print("[task1] building tables/plots ...")
    os.makedirs(out_dir, exist_ok=True)
    models = list(d["clean_baseline"].keys())

    # ---- Table: clean baseline (+ CLIP zero-shot) ----
    rows = []
    for m in models:
        r = d["clean_baseline"][m]
        rows.append({"model": m, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                     "mean_max_conf": r["mean_max_conf"]})
    rows.append({"model": "clip_zero_shot", **d["clip_zero_shot_baseline"]})
    df_clean = pd.DataFrame(rows).set_index("model")
    print("\n=== Clean baseline ===")
    print(df_clean.round(4))
    df_clean.to_csv(os.path.join(out_dir, "task1_clean_baseline.csv"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df_clean.index, df_clean["accuracy"], color="#4C72B0")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20)
    savefig(fig, out_dir, "Task 1 - Clean baseline accuracy")

    # ---- Color bias ----
    color_rows = []
    for cond in ["grayscale", "hue_rotate_90"]:
        for m in models:
            r = d["color_bias"][cond][m]
            color_rows.append({"condition": cond, "model": m, **r})
    df_color = pd.DataFrame(color_rows)
    print("\n=== Color bias ===")
    print(df_color.round(4).to_string(index=False))
    df_color.to_csv(os.path.join(out_dir, "task1_color_bias.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x = np.arange(len(models))
    gray = [d["color_bias"]["grayscale"][m]["accuracy"] for m in models]
    hue = [d["color_bias"]["hue_rotate_90"][m]["accuracy"] for m in models]
    clean = [d["clean_baseline"][m]["accuracy"] for m in models]
    ax.bar(x - width / 2, gray, width, label="grayscale")
    ax.bar(x + width / 2, hue, width, label="hue rotate 90")
    ax.plot(x, clean, "ko--", label="clean", markersize=5)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Accuracy")
    ax.legend()
    savefig(fig, out_dir, "Task 1 - Color-bias interventions vs clean accuracy")

    # ---- Color bias: prediction consistency vs clean ----
    fig, ax = plt.subplots(figsize=(7, 4))
    gray_c = [d["color_bias"]["grayscale"][m]["consistency_vs_clean"] for m in models]
    hue_c = [d["color_bias"]["hue_rotate_90"][m]["consistency_vs_clean"] for m in models]
    ax.bar(x - width / 2, gray_c, width, label="grayscale")
    ax.bar(x + width / 2, hue_c, width, label="hue rotate 90")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Prediction consistency vs clean")
    ax.legend()
    savefig(fig, out_dir, "Task 1 - Color-bias prediction consistency vs clean")

    # ---- Shape vs texture bias ----
    st_rows = []
    for m in models:
        r = d["shape_texture_bias"][m]
        st_rows.append({"model": m, **r})
    df_st = pd.DataFrame(st_rows).set_index("model")
    print("\n=== Shape vs texture bias ===")
    print(df_st.round(2))
    df_st.to_csv(os.path.join(out_dir, "task1_shape_texture_bias.csv"))

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(df_st.index, df_st["shape_bias_pct"], color="#55A868", label="shape bias %")
    ax1.set_ylabel("Shape bias (%)"); ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.plot(df_st.index, df_st["coverage_pct"], "ro-", label="coverage %")
    ax2.set_ylabel("Coverage (%)"); ax2.set_ylim(0, 100)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.xticks(rotation=20)
    savefig(fig, out_dir, "Task 1 - Shape bias and cue-conflict coverage by model")

    # ---- Translation curve ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in models:
        mags = sorted(int(k) for k in d["translation_curve"][m].keys())
        acc = [d["translation_curve"][m][str(mg)]["accuracy"] for mg in mags]
        cons = [d["translation_curve"][m][str(mg)]["consistency"] for mg in mags]
        axes[0].plot(mags, acc, "o-", label=m)
        axes[1].plot(mags, cons, "o-", label=m)
    axes[0].set_xlabel("Translation (px)"); axes[0].set_ylabel("Accuracy")
    axes[1].set_xlabel("Translation (px)"); axes[1].set_ylabel("Consistency vs clean")
    axes[0].legend(); axes[1].legend()
    savefig(fig, out_dir, "Task 1 - Accuracy and consistency vs translation magnitude")

    # ---- Patch shuffle ----
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models))
    clean_acc = [d["clean_baseline"][m]["accuracy"] for m in models]
    shuf_acc = [d["patch_shuffle"][m]["accuracy"] for m in models]
    cons = [d["patch_shuffle"][m]["consistency_vs_clean"] for m in models]
    width = 0.35
    ax.bar(x - width / 2, clean_acc, width, label="clean")
    ax.bar(x + width / 2, shuf_acc, width, label="patch-shuffled")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Accuracy")
    ax.legend()
    for i, c in enumerate(cons):
        ax.annotate(f"cons={c:.2f}", (i, shuf_acc[i]), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8)
    savefig(fig, out_dir, "Task 1 - Patch shuffle accuracy drop (labels show consistency)")

    # ---- Representation stability heatmap ----
    conditions = list(next(iter(d["representation_stability"].values())).keys())
    mat = np.array([[d["representation_stability"][m][c] for c in conditions] for m in models])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(conditions))); ax.set_xticklabels(conditions, rotation=20)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                     color="white" if mat[i, j] < 0.6 else "black", fontsize=9)
    fig.colorbar(im, label="cosine stability I_T")
    savefig(fig, out_dir, "Task 1 - Representation stability I_T by model and condition")

    print(f"\n[task1] cue-conflict images: accepted={d['cue_conflict_counts']['accepted']} "
          f"rejected={d['cue_conflict_counts']['rejected']}")

    # ---- Cue-conflict accept/reject counts ----
    fig, ax = plt.subplots(figsize=(4, 4))
    counts = d["cue_conflict_counts"]
    ax.bar(list(counts.keys()), list(counts.values()), color=["#55A868", "#C44E52"])
    ax.set_ylabel("Count")
    savefig(fig, out_dir, "Task 1 - Cue-conflict generation accept vs reject counts")


# =============================================================================
# Task 2
# =============================================================================

def _dominant_confusions_plot(confusions, out_dir, title, k=5):
    if not confusions:
        return
    cdf = pd.DataFrame(confusions[:k])
    cdf["pair"] = cdf["true_class"] + " -> " + cdf["predicted_class"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(cdf["pair"][::-1], cdf["count"][::-1], color="#C44E52")
    ax.set_xlabel("Count")
    savefig(fig, out_dir, title)


def _per_domain_grouped_bar(comparison_table, methods, out_dir, title, metric="accuracy"):
    domains = list(next(iter(comparison_table.values()))["per_domain_val"].keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(domains))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [comparison_table[m]["per_domain_val"][dom][metric] for dom in domains]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m)
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.legend()
    savefig(fig, out_dir, title)


def task2_tables_and_plots(d, out_dir):
    if d is None:
        print("[task2] no summary found, skipping")
        return
    print("[task2] building tables/plots ...")
    os.makedirs(out_dir, exist_ok=True)
    methods = list(d["comparison_table"].keys())

    rows = []
    for m in methods:
        r = d["comparison_table"][m]
        rows.append({
            "method": m, "mean_source_acc": r["mean_source_acc"],
            "target_accuracy": r["target_accuracy"], "target_macro_f1": r["target_macro_f1"],
            "target_acc_change_vs_source_only": r["target_acc_change_vs_source_only"],
            "domain_separability": r["domain_separability"],
        })
    df = pd.DataFrame(rows).set_index("method")
    print("\n=== Task 2 comparison table ===")
    print(df.round(4))
    df.to_csv(os.path.join(out_dir, "task2_comparison.csv"))

    # ---- Target accuracy by method ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df.index, df["target_accuracy"], color="#C44E52")
    ax.axhline(df.loc["source_only", "target_accuracy"], color="k", linestyle="--",
               label="source-only baseline")
    ax.set_ylabel("Target (Sketch) accuracy")
    ax.legend()
    savefig(fig, out_dir, "Task 2 - Target Sketch accuracy by method")

    # ---- Mean source acc vs target acc (the "did adaptation cost source performance" plot) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width / 2, df["mean_source_acc"], width, label="mean source acc")
    ax.bar(x + width / 2, df["target_accuracy"], width, label="target acc")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.legend()
    savefig(fig, out_dir, "Task 2 - Mean source accuracy vs target accuracy")

    # ---- Per-domain source accuracy / macro-F1 by method ----
    _per_domain_grouped_bar(d["comparison_table"], methods, out_dir,
                             "Task 2 - Per-domain source accuracy by method", metric="accuracy")
    _per_domain_grouped_bar(d["comparison_table"], methods, out_dir,
                             "Task 2 - Per-domain source macro-F1 by method", metric="macro_f1")

    # ---- Domain separability vs target accuracy scatter (the key diagnostic plot) ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["domain_separability"], df["target_accuracy"], s=80)
    for m in methods:
        ax.annotate(m, (df.loc[m, "domain_separability"], df.loc[m, "target_accuracy"]),
                    textcoords="offset points", xytext=(6, 4))
    ax.axvline(0.5, color="gray", linestyle=":", label="chance separability (0.5)")
    ax.set_xlabel("Domain separability (source vs target)")
    ax.set_ylabel("Target accuracy")
    ax.legend()
    savefig(fig, out_dir, "Task 2 - Domain separability vs target accuracy")

    # ---- Training curves (cls_loss), log-scale y since DANN/CDAN are unstable ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        curve = d["training_curves"][m]
        epochs = [c["epoch"] for c in curve]
        losses = [c["cls_loss"] for c in curve]
        ax.plot(epochs, losses, "o-", label=m)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Classification loss (log scale)")
    ax.legend()
    savefig(fig, out_dir, "Task 2 - Training curves classification loss (log scale)")

    # ---- DANN/CDAN: domain loss + GRL alpha schedule (instability diagnostics) ----
    for m in ["dann", "cdan"]:
        curve = d["training_curves"].get(m, [])
        if not curve or "domain_loss" not in curve[0]:
            continue
        epochs = [c["epoch"] for c in curve]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(epochs, [c["cls_loss"] for c in curve], "o-", label="cls_loss")
        axes[0].plot(epochs, [c["domain_loss"] for c in curve], "s-", label="domain_loss")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log scale)")
        axes[0].legend()
        axes[1].plot(epochs, [c["grl_alpha"] for c in curve], "o-", color="#8172B2")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("GRL alpha")
        savefig(fig, out_dir, f"Task 2 - {m.upper()} loss and GRL alpha schedule")

    # ---- Per-class delta vs source-only, for each non-baseline method ----
    for m in ["dan", "dann", "cdan"]:
        if m not in d["class_analysis_vs_source_only"]:
            continue
        changes = d["class_analysis_vs_source_only"][m]["per_class_change"]
        cdf = pd.DataFrame(changes).sort_values("delta")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#C44E52" if v < 0 else "#55A868" for v in cdf["delta"]]
        ax.barh(cdf["class_name"], cdf["delta"], color=colors)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Accuracy change vs source-only")
        savefig(fig, out_dir, f"Task 2 - Per-class target accuracy change {m}")

        # ---- Dominant confusions for the same method ----
        confusions = d["class_analysis_vs_source_only"][m].get("dominant_confusions", [])
        _dominant_confusions_plot(confusions, out_dir,
                                   f"Task 2 - Dominant Sketch confusions {m}")

    # Flag potential instability automatically
    for m in ["dann", "cdan"]:
        if m in d["training_curves"]:
            losses = [c["cls_loss"] for c in d["training_curves"][m]]
            if max(losses) > 50 * losses[0]:
                print(f"  [WARNING] {m}: cls_loss grew from {losses[0]:.3f} to "
                      f"{max(losses):.1f} -- looks like training instability, not just "
                      f"'excessive domain confusion'. Check dominant_confusions for that "
                      f"method: if almost everything is predicted as one class, this is "
                      f"mode collapse, not the intended negative-transfer story.")


# =============================================================================
# Task 3
# =============================================================================

def task3_tables_and_plots(d, out_dir):
    if d is None:
        print("[task3] no summary found, skipping")
        return
    print("[task3] building tables/plots ...")
    os.makedirs(out_dir, exist_ok=True)
    methods = list(d["comparison_table"].keys())

    rows = []
    for m in methods:
        r = d["comparison_table"][m]
        rows.append({
            "method": m, "mean_source_accuracy": r["mean_source_accuracy"],
            "worst_source_accuracy": r["worst_source_accuracy"],
            "sketch_accuracy": r["sketch_accuracy"],
            "sketch_accuracy_change_vs_erm": r["sketch_accuracy_change_vs_erm"],
            "source_domain_separability": r["source_domain_separability"],
            "sharpness_proxy": r["sharpness_proxy"],
        })
    df = pd.DataFrame(rows).set_index("method")
    print("\n=== Task 3 comparison table ===")
    print(df.round(4))
    df.to_csv(os.path.join(out_dir, "task3_comparison.csv"))

    # ---- Mean/worst source vs Sketch accuracy ----
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods)); width = 0.25
    ax.bar(x - width, df["mean_source_accuracy"], width, label="mean source acc")
    ax.bar(x, df["worst_source_accuracy"], width, label="worst source acc")
    ax.bar(x + width, df["sketch_accuracy"], width, label="Sketch (unseen) acc")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.legend()
    savefig(fig, out_dir, "Task 3 - Source mean and worst accuracy vs unseen-domain accuracy")

    # ---- Per-domain source accuracy / macro-F1 by method ----
    _per_domain_grouped_bar(d["comparison_table"], methods, out_dir,
                             "Task 3 - Per-domain source accuracy by method", metric="accuracy")
    _per_domain_grouped_bar(d["comparison_table"], methods, out_dir,
                             "Task 3 - Per-domain source macro-F1 by method", metric="macro_f1")

    # ---- Sharpness proxy ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df.index, df["sharpness_proxy"], color="#8172B2")
    ax.set_ylabel("Delta_sharp (lower = locally flatter)")
    savefig(fig, out_dir, "Task 3 - Local sharpness proxy by method")

    # ---- Source-domain separability (chance = 33.3%) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df.index, df["source_domain_separability"], color="#CCB974")
    ax.axhline(1 / 3, color="k", linestyle="--", label="chance (33.3%)")
    ax.set_ylabel("Source-domain separability")
    ax.legend()
    savefig(fig, out_dir, "Task 3 - Source-domain separability Photo Art Cartoon")

    # ---- Sharpness vs Sketch accuracy scatter (does flatter -> better generalization?) ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["sharpness_proxy"], df["sketch_accuracy"], s=80)
    for m in methods:
        ax.annotate(m, (df.loc[m, "sharpness_proxy"], df.loc[m, "sketch_accuracy"]),
                    textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Sharpness proxy (lower = flatter)")
    ax.set_ylabel("Sketch accuracy")
    savefig(fig, out_dir, "Task 3 - Local flatness vs unseen-domain accuracy")

    # ---- Training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if "dan_dg" in d["training_curves"]:
        curve = d["training_curves"]["dan_dg"]
        epochs = [c["epoch"] for c in curve]
        axes[0].plot(epochs, [c["cls_loss"] for c in curve], "o-", label="cls_loss")
        axes[0].plot(epochs, [c["pairwise_mmd"] for c in curve], "s-", label="pairwise_mmd")
        axes[0].legend()
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
    if "sam" in d["training_curves"]:
        curve = d["training_curves"]["sam"]
        epochs = [c["epoch"] for c in curve]
        axes[1].plot(epochs, [c["cls_loss"] for c in curve], "o-", color="#55A868",
                      label="cls_loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
    savefig(fig, out_dir, "Task 3 - DAN-DG and SAM training curves")

    # ---- Per-class delta vs ERM (+ dominant confusions) ----
    for m in ["dan_dg", "sam"]:
        if m not in d["class_analysis_vs_erm"]:
            continue
        changes = d["class_analysis_vs_erm"][m]["per_class_change"]
        cdf = pd.DataFrame(changes).sort_values("delta")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#C44E52" if v < 0 else "#55A868" for v in cdf["delta"]]
        ax.barh(cdf["class_name"], cdf["delta"], color=colors)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Sketch accuracy change vs ERM")
        savefig(fig, out_dir, f"Task 3 - Per-class Sketch accuracy change {m}")

        confusions = d["class_analysis_vs_erm"][m].get("dominant_confusions", [])
        _dominant_confusions_plot(confusions, out_dir,
                                   f"Task 3 - Dominant Sketch confusions {m}")


# =============================================================================
# Task 4
# =============================================================================

def task4_tables_and_plots(d, out_dir):
    if d is None:
        print("[task4] no summary found yet, skipping (this is expected for now)")
        return
    print("[task4] building tables/plots ...")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Table 1: score comparison on frozen Vanilla ----
    rows = []
    for score_name, r in d["vanilla_score_comparison"].items():
        rows.append({"score": score_name, **r})
    df1 = pd.DataFrame(rows).set_index("score")
    print("\n=== Task 4 Table 1: score comparison (frozen Vanilla) ===")
    print(df1.round(4))
    df1.to_csv(os.path.join(out_dir, "task4_score_comparison.csv"))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df1))
    width = 0.35
    ax.bar(x - width / 2, df1["auroc_near"], width, label="AUROC (near)")
    ax.bar(x + width / 2, df1["auroc_far"], width, label="AUROC (far)")
    ax.set_xticks(x); ax.set_xticklabels(df1.index)
    ax.axhline(0.5, color="k", linestyle=":", label="chance")
    ax.set_ylabel("AUROC")
    ax.legend()
    savefig(fig, out_dir, "Task 4 - Post-hoc score AUROC near vs far unknowns")

    # ---- FPR@95%TPR comparison (the "cost of a 95% known-accept policy" plot) ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, df1["fpr95_near"], width, label="FPR@95TPR (near)")
    ax.bar(x + width / 2, df1["fpr95_far"], width, label="FPR@95TPR (far)")
    ax.set_xticks(x); ax.set_xticklabels(df1.index)
    ax.set_ylabel("False positive rate at 95% known-acceptance")
    ax.legend()
    savefig(fig, out_dir, "Task 4 - FPR at 95pct TPR near vs far unknowns")

    # ---- Table 2: model comparison ----
    rows = []
    for model_name, r in d["model_comparison"].items():
        rows.append({"model": model_name, **r})
    df2 = pd.DataFrame(rows).set_index("model")
    print("\n=== Task 4 Table 2: Vanilla/GCSC/PROSER comparison ===")
    print(df2.round(4))
    df2.to_csv(os.path.join(out_dir, "task4_model_comparison.csv"))

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(df2.index, df2["csa"], color="#4C72B0", alpha=0.6, label="CSA (closed-set acc)")
    ax1.set_ylabel("Closed-set accuracy")
    ax2 = ax1.twinx()
    ax2.plot(df2.index, df2["auroc_all"], "ro-", label="AUROC (all unknowns)")
    ax2.set_ylabel("AUROC")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.xticks(rotation=15)
    savefig(fig, out_dir, "Task 4 - Closed-set accuracy vs open-set AUROC by model")

    # ---- Near vs far AUROC per model (does the model/score generalize to far-OOD?) ----
    fig, ax = plt.subplots(figsize=(7, 4))
    x2 = np.arange(len(df2))
    ax.bar(x2 - width / 2, df2["auroc_near"], width, label="AUROC (near)")
    ax.bar(x2 + width / 2, df2["auroc_far"], width, label="AUROC (far)")
    ax.set_xticks(x2); ax.set_xticklabels(df2.index, rotation=15)
    ax.axhline(0.5, color="k", linestyle=":", label="chance")
    ax.set_ylabel("AUROC")
    ax.legend()
    savefig(fig, out_dir, "Task 4 - Model comparison AUROC near vs far unknowns")

    # ---- Failure analysis table (+ score margin below threshold bar chart) ----
    for group in ["near", "far"]:
        fails = d["failure_analysis"].get(group, [])
        if not fails:
            continue
        print(f"\n=== Task 4 failure analysis ({group}) ===")
        fdf = pd.DataFrame(fails)
        print(fdf.to_string(index=False))
        fdf.to_csv(os.path.join(out_dir, f"task4_failure_analysis_{group}.csv"), index=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [f"{r.unknown_class}\n-> {r.predicted_known_class}" for r in fdf.itertuples()]
        margin = fdf["threshold"] - fdf["score"]
        ax.barh(labels, margin, color="#C44E52")
        ax.set_xlabel("Score margin below threshold (accepted-as-known unknowns)")
        savefig(fig, out_dir, f"Task 4 - Accepted unknown failure cases {group}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True,
                         help="Folder containing task{1,2,3,4}_summary.json")
    parser.add_argument("--out_dir", default="./figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d1 = load_json(os.path.join(args.results_dir, "task1_summary.json"))
    d2 = load_json(os.path.join(args.results_dir, "task2_summary.json"))
    d3 = load_json(os.path.join(args.results_dir, "task3_summary.json"))
    d4 = load_json(os.path.join(args.results_dir, "task4_summary.json"))

    task1_tables_and_plots(d1, os.path.join(args.out_dir, "task1"))
    task2_tables_and_plots(d2, os.path.join(args.out_dir, "task2"))
    task3_tables_and_plots(d3, os.path.join(args.out_dir, "task3"))
    task4_tables_and_plots(d4, os.path.join(args.out_dir, "task4"))

    print(f"\nAll done. Figures and CSVs under {args.out_dir}/")


if __name__ == "__main__":
    main()
