"""
UMAP visualization: for each backbone, fit ONE 2D projection to the combined
clean + transformed features (per condition) so both conditions live in the
same space. Color = ground-truth class, marker = clean vs transformed.

Uses UMAP (per user's choice) rather than t-SNE. Keep the image subset and
random seed fixed across backbones/conditions so plots are comparable in
spirit, though absolute coordinates should NOT be compared across backbones
(each has its own separately-fit projection).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm

SEED = 6304
MARKERS = {"clean": "o", "transformed": "^"}
CMAP = plt.get_cmap("tab10")


def fit_umap_projection(clean_feats, transformed_feats, seed=SEED, n_neighbors=15, min_dist=0.1):
    """clean_feats, transformed_feats: (N, D) numpy arrays, paired index-for-index.
    Fits UMAP jointly on the concatenation of both so they land in one shared space."""
    combined = np.concatenate([clean_feats, transformed_feats], axis=0)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=seed, metric="cosine")
    embedding = reducer.fit_transform(combined)
    n = clean_feats.shape[0]
    return embedding[:n], embedding[n:]  # (clean_2d, transformed_2d)


def plot_umap(clean_2d, transformed_2d, labels, backbone_name, condition_name, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(clean_2d[mask, 0], clean_2d[mask, 1],
                   color=CMAP(cls % 10), marker=MARKERS["clean"],
                   alpha=0.7, s=25, label=f"class {cls} (clean)" if False else None)
        ax.scatter(transformed_2d[mask, 0], transformed_2d[mask, 1],
                   color=CMAP(cls % 10), marker=MARKERS["transformed"],
                   alpha=0.7, s=25)

    # Custom legend: colors = classes, marker shapes = condition.
    from matplotlib.lines import Line2D
    class_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=CMAP(c % 10),
                             markersize=8, label=f"class {c}") for c in np.unique(labels)]
    marker_handles = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, label="clean"),
        Line2D([0], [0], marker='^', color='gray', linestyle='None', markersize=8, label="transformed"),
    ]
    ax.legend(handles=class_handles + marker_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title(f"{backbone_name} -- {condition_name} (UMAP, seed={SEED})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_all_umap_plots(models, clean_feats_by_model, transformed_feats_by_condition_by_model,
                        labels, out_dir="task1/results/umap"):
    """
    clean_feats_by_model: {model_name: (N, D) numpy array}
    transformed_feats_by_condition_by_model: {model_name: {condition: (N, D) numpy array}}
    """
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(name, cond) for name in models for cond in transformed_feats_by_condition_by_model[name]]
    for name, condition in tqdm(jobs, desc="Fitting UMAP projections", unit="plot"):
        clean_feats = clean_feats_by_model[name]
        trans_feats = transformed_feats_by_condition_by_model[name][condition]
        clean_2d, trans_2d = fit_umap_projection(clean_feats, trans_feats)
        out_path = os.path.join(out_dir, f"{name}_{condition}_umap.png")
        plot_umap(clean_2d, trans_2d, labels, name, condition, out_path)
        tqdm.write(f"Saved {out_path}")
