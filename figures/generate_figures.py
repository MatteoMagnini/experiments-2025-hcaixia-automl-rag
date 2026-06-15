"""Regenerate the paper figures from the AutoML-RAG experiment results.

Inputs:
  - incumbents.csv_results.csv : all configurations evaluated by SMAC3
  - incumbents.csv             : the Pareto-front incumbents

Outputs (figures/):
  - the original figures of the paper (Pareto front, accuracy boxplots, KDEs)
    rebuilt on the new data, plus new figures for the generation model and
    the BERTScore-F1 (gold) objective.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
DATA = ROOT.parent / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# large fonts so the figures stay readable when scaled to (half-)column width
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
})

EMBEDDER_SHORT = {
    "nomic-embed-text-v2-moe": "nomic-v2-moe",
    "qwen3-embedding-0.6b": "qwen3-0.6b",
    "qwen3-embedding-4b": "qwen3-4b",
    "granite-embedding-107m": "granite-107m",
    "granite-embedding-278m": "granite-278m",
    "bert-base-italian-xxl-cased": "bert-xxl (it)",
}

GEN_MODEL_SHORT = {
    "google/gemma-3-4b-it": "gemma-3-4b",
    "google/gemma-3-12b-it": "gemma-3-12b",
    "google/gemma-3-27b-it": "gemma-3-27b",
    "qwen/qwen3-8b": "qwen3-8b",
    "qwen/qwen3-14b": "qwen3-14b",
    "qwen/qwen3-32b": "qwen3-32b",
}


def load_data():
    df = pd.read_csv(DATA / "incumbents.csv_results.csv")
    df = df[df["status"] != "failed"].copy()
    df["accuracy"] = 1 - df["1 - accuracy"]
    df["number of documents"] = df["number of documents"].astype(int)
    df["embedder_short"] = df["embedder"].map(EMBEDDER_SHORT)
    df["gen_model_short"] = df["gen_model"].map(GEN_MODEL_SHORT)

    inc = pd.read_csv(DATA / "incumbents.csv")
    # crashed evaluations get a placeholder loss of 1.0 in the incumbents file
    inc.loc[inc["1 - bert_f1_gold"] >= 1.0, "1 - bert_f1_gold"] = pd.NA
    inc["accuracy"] = 1 - inc["1 - accuracy"]
    inc["bert_f1_gold"] = 1 - inc["1 - bert_f1_gold"]
    inc["number of documents"] = inc["number of documents"].astype(int)
    inc = inc.sort_values("1 - accuracy").reset_index(drop=True)
    inc["id"] = [f"I{i + 1}" for i in range(len(inc))]
    inc["label"] = [
        f"{i}: {EMBEDDER_SHORT[e]} | {r} | {GEN_MODEL_SHORT[g]} | {d} docs"
        + ("" if pd.notna(b) else " (no BERT)")
        for i, e, r, g, d, b in zip(inc["id"], inc["embedder"],
                                    inc["retriever"], inc["gen_model"],
                                    inc["number of documents"],
                                    inc["bert_f1_gold"])
    ]
    return df, inc


def save(fig, name, tight=True):
    # the tight bbox crops mplot3d axis labels, so 3D figures skip it
    fig.savefig(FIGURES / name, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def front_2d(points, x, y):
    """Non-dominated subset (both objectives minimized) of a 2D projection."""
    points = points.sort_values([x, y])
    best = float("inf")
    keep = []
    for _, row in points.iterrows():
        if row[y] < best:
            keep.append(row)
            best = row[y]
    return pd.DataFrame(keep)


def pareto_scatter(df, inc, x, y, xlabel, ylabel, name):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    data = df.dropna(subset=[x, y])
    ax.scatter(data[x], data[y], marker="x", color="tab:blue")
    incumbents = inc.dropna(subset=[x, y])
    # the incumbents are Pareto-optimal in 3D; connect only the ones that are
    # also non-dominated in this 2D projection
    front = front_2d(incumbents, x, y)
    ax.plot(front[x], front[y], linestyle=":", color="tab:red", linewidth=1)
    ax.scatter(incumbents[x], incumbents[y], marker="x", color="red", zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save(fig, name)


def boxplot(df, x, y, xlabel, ylabel, name, hue=None, rotation=45,
            order=None, tick_every=None, figsize=(6, 4), legend_top=False):
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x, y=y, hue=hue, order=order, ax=ax)
    if legend_top:
        # one row above the axes, clear of the boxes
        sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.02),
                        ncol=4, title=None, frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    if tick_every is not None:
        # with many categories the labels collide: keep one every tick_every
        for i, label in enumerate(ax.get_xticklabels()):
            if i != 0 and (i + 1) % tick_every:
                label.set_visible(False)
    save(fig, name)


def paired_boxplots(df, y, ylabel, name, docs_order, emb_order):
    """Number-of-documents and embedder/retriever boxplots as one figure, so
    the two panels share height, font scale, and comparable box widths.

    The width ratio matches the number of box slots per panel (20 document
    categories vs. 6 embedders x 4 retrievers).
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.5), sharey=True,
        gridspec_kw=dict(width_ratios=[1, 1.2], wspace=0.05))

    # neutral gray: the retriever palette of the right panel does not apply
    sns.boxplot(data=df, x="number of documents", y=y, order=docs_order,
                color="0.75", ax=ax1)
    ax1.set_xlabel("Number of Documents")
    ax1.set_ylabel(ylabel)
    for i, label in enumerate(ax1.get_xticklabels()):
        if i != 0 and (i + 1) % 5:
            label.set_visible(False)

    sns.boxplot(data=df, x="embedder_short", y=y, hue="retriever",
                order=emb_order, ax=ax2)
    sns.move_legend(ax2, "lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=4, title=None, frameon=False)
    ax2.set_xlabel("Embedder")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=30)
    save(fig, name)


def paired_binned_boxplots(df, y, ylabel, name, bins=6):
    """Chunk-length and overlap-percentage boxplots as one figure with a
    shared y axis and uniform style, mirroring paired_boxplots."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True,
                             gridspec_kw=dict(wspace=0.05))
    panels = [("chunk_token_length", "Chunk Length Bin", "{:.0f}"),
              ("overlap_percentage", "Overlap Percentage Bin", "{:.2f}")]
    for ax, (x, xlabel, fmt) in zip(axes, panels):
        binned = df.copy()
        cut = pd.cut(binned[x], bins=bins)
        labels = {c: f"({fmt.format(c.left)}, {fmt.format(c.right)}]"
                  for c in cut.cat.categories}
        binned["bin"] = cut.map(labels)
        order = list(labels.values())
        sns.boxplot(data=binned, x="bin", y=y, order=order, color="0.75",
                    ax=ax)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel("")
    save(fig, name)


def binned_boxplot(df, x, y, bins, xlabel, ylabel, name):
    binned = df.copy()
    cut = pd.cut(binned[x], bins=bins, precision=3)
    binned["bin"] = cut.astype(str)
    order = [str(c) for c in cut.cat.categories]
    boxplot(binned, "bin", y, xlabel, ylabel, name, order=order)


def kde(df, x, y, xlabel, ylabel, name):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data=df, x=x, y=y, fill=True, levels=10, ax=ax)
    sns.kdeplot(
        data=df, x=x, y=y, levels=10, color="white",
        linewidths=0.5, linestyles="--", ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save(fig, name)


OBJECTIVES = ["1 - accuracy", "number of documents", "1 - bert_f1_gold"]
OBJECTIVE_LABELS = ["1 - Accuracy", "Number of Documents", "1 - BERTScore F1"]


def scatter_3d(df, inc, name):
    """Static 3D view of the objective space with the incumbent surface."""
    data = df.dropna(subset=OBJECTIVES)
    front = inc.dropna(subset=OBJECTIVES)
    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*(data[c] for c in OBJECTIVES), marker="x", s=35,
               color="tab:blue", alpha=0.55, label="Evaluated",
               depthshade=False)
    # translucent Pareto surface through the incumbents
    ax.plot_trisurf(front[OBJECTIVES[0]], front[OBJECTIVES[1]],
                    front[OBJECTIVES[2]], color="red", alpha=0.18,
                    edgecolor="darkred", linewidth=0.4)
    ax.scatter(*(front[c] for c in OBJECTIVES), marker="x", s=90,
               linewidths=2.5, color="red", label="Incumbents",
               depthshade=False)
    for _, row in front.iterrows():
        ax.text(row[OBJECTIVES[0]], row[OBJECTIVES[1]],
                row[OBJECTIVES[2]] + 0.003, row["id"], fontsize=13,
                color="darkred", fontweight="bold")
    # shadows on the floor to make depth readable
    ax.scatter(data[OBJECTIVES[0]], data[OBJECTIVES[1]],
               zs=data[OBJECTIVES[2]].min(), marker=".",
               color="gray", alpha=0.25, s=8)
    ax.set_xlabel(OBJECTIVE_LABELS[0], fontsize=16, labelpad=10)
    ax.set_ylabel(OBJECTIVE_LABELS[1], fontsize=16, labelpad=10)
    ax.set_zlabel("1 - BERT F1", fontsize=16, labelpad=10)
    ax.tick_params(labelsize=12)
    ax.view_init(elev=22, azim=40)
    ax.legend(loc="upper left", fontsize=12)
    fig.subplots_adjust(left=0.0, right=0.62)
    fig.text(0.63, 0.5, "Incumbents\n" + "\n".join(inc["label"]),
             va="center", fontsize=8, family="monospace")
    save(fig, name, tight=False)


def scatter_3d_interactive(df, inc, name):
    """Rotatable plotly version of the 3D objective space with config hover."""
    import plotly.graph_objects as go

    def hover(d):
        return [
            f"embedder: {emb}<br>retriever: {ret}<br>gen model: {gen}"
            f"<br>chunk: {chunk}<br>overlap: {over:.2f}"
            f"<br>accuracy: {1 - acc:.2f}<br>BERT F1: {1 - bert:.3f}"
            for emb, ret, gen, chunk, over, acc, bert in zip(
                d["embedder"], d["retriever"], d["gen_model"],
                d["chunk_token_length"], d["overlap_percentage"],
                d["1 - accuracy"], d["1 - bert_f1_gold"])
        ]

    data = df.dropna(subset=OBJECTIVES)
    front = inc.dropna(subset=OBJECTIVES)
    fig = go.Figure([
        go.Scatter3d(
            x=data[OBJECTIVES[0]], y=data[OBJECTIVES[1]],
            z=data[OBJECTIVES[2]], mode="markers", name="Evaluated",
            marker=dict(size=4, symbol="x", color="steelblue", opacity=0.75),
            text=hover(data), hoverinfo="text",
        ),
        go.Mesh3d(
            x=front[OBJECTIVES[0]], y=front[OBJECTIVES[1]],
            z=front[OBJECTIVES[2]], color="red", opacity=0.15,
            alphahull=-1, name="Pareto surface", hoverinfo="skip",
        ),
        go.Scatter3d(
            x=front[OBJECTIVES[0]], y=front[OBJECTIVES[1]],
            z=front[OBJECTIVES[2]], mode="markers+text", name="Incumbents",
            marker=dict(size=6, symbol="x", color="red"),
            text=front["id"], textposition="top center",
            textfont=dict(color="darkred", size=11),
            hovertext=hover(front), hoverinfo="text",
        ),
    ])
    fig.update_layout(
        scene=dict(xaxis_title=OBJECTIVE_LABELS[0],
                   yaxis_title=OBJECTIVE_LABELS[1],
                   zaxis_title=OBJECTIVE_LABELS[2]),
    )
    fig.write_html(FIGURES / name, include_plotlyjs="cdn")
    print(f"  wrote figures/{name}")


def surface_grid(front, resolution=60):
    """Smooth Pareto surface interpolated through the incumbents."""
    import numpy as np
    from scipy.interpolate import griddata

    x, y, z = (front[c].to_numpy(dtype=float) for c in OBJECTIVES)
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xg, yg = np.meshgrid(xi, yi)
    zg = griddata((x, y), z, (xg, yg), method="linear")
    return xg, yg, zg


def pareto_surface_3d(df, inc, name):
    """3D objective space with the incumbents rendered as a smooth surface."""
    import numpy as np

    data = df.dropna(subset=OBJECTIVES)
    front = inc.dropna(subset=OBJECTIVES)
    xg, yg, zg = surface_grid(front)

    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xg, yg, np.ma.masked_invalid(zg), cmap="viridis",
                           alpha=0.85, linewidth=0, antialiased=True)
    ax.scatter(*(data[c] for c in OBJECTIVES), marker="x", s=30,
               color="gray", alpha=0.45, label="Evaluated", depthshade=False)
    ax.scatter(*(front[c] for c in OBJECTIVES), marker="o", s=55,
               color="red", edgecolor="black", label="Incumbents",
               depthshade=False, zorder=5)
    for _, row in front.iterrows():
        ax.text(row[OBJECTIVES[0]], row[OBJECTIVES[1]],
                row[OBJECTIVES[2]] + 0.004, row["id"], fontsize=9,
                color="darkred", fontweight="bold")
    ax.set_xlabel(OBJECTIVE_LABELS[0])
    ax.set_ylabel(OBJECTIVE_LABELS[1])
    ax.set_zlabel("1 - BERT F1", labelpad=6)
    ax.view_init(elev=24, azim=40)
    ax.legend(loc="upper left")
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="1 - BERT F1")
    fig.subplots_adjust(left=0.04, right=0.62)
    fig.text(0.66, 0.5, "Incumbents\n" + "\n".join(inc["label"]),
             va="center", fontsize=8, family="monospace")
    save(fig, name, tight=False)


def pareto_surface_3d_interactive(df, inc, name):
    """Rotatable plotly version of the smooth incumbent surface."""
    import plotly.graph_objects as go

    data = df.dropna(subset=OBJECTIVES)
    front = inc.dropna(subset=OBJECTIVES)
    xg, yg, zg = surface_grid(front)

    fig = go.Figure([
        go.Surface(
            x=xg, y=yg, z=zg, colorscale="Viridis", opacity=0.85,
            name="Pareto surface", showscale=True,
            colorbar=dict(title="1 - BERT F1", len=0.6),
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=data[OBJECTIVES[0]], y=data[OBJECTIVES[1]],
            z=data[OBJECTIVES[2]], mode="markers", name="Evaluated",
            marker=dict(size=3.5, symbol="x", color="gray", opacity=0.6),
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=front[OBJECTIVES[0]], y=front[OBJECTIVES[1]],
            z=front[OBJECTIVES[2]], mode="markers+text", name="Incumbents",
            marker=dict(size=6, color="red", line=dict(color="black", width=1)),
            text=front["id"], textposition="top center",
            textfont=dict(color="darkred", size=11),
            hovertext=front["label"], hoverinfo="text",
        ),
    ])
    fig.update_layout(
        scene=dict(xaxis_title=OBJECTIVE_LABELS[0],
                   yaxis_title=OBJECTIVE_LABELS[1],
                   zaxis_title="1 - BERT F1"),
    )
    fig.write_html(FIGURES / name, include_plotlyjs="cdn")
    print(f"  wrote figures/{name}")


def pareto_colored_by_bert(df, inc, name):
    """The original Pareto view with the third objective encoded as color."""
    data = df.dropna(subset=OBJECTIVES)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sc = ax.scatter(data["1 - accuracy"], data["number of documents"],
                    c=data["bert_f1_gold"], cmap="viridis", s=45,
                    edgecolors="black", linewidths=0.3)
    front = inc.dropna(subset=OBJECTIVES)
    ax.scatter(front["1 - accuracy"], front["number of documents"],
               facecolors="none", edgecolors="red", s=140, linewidths=1.5,
               label="Incumbents")
    fig.colorbar(sc, ax=ax, label="BERTScore F1")
    ax.set_xlabel("1 - Accuracy")
    ax.set_ylabel("Number of Documents")
    ax.legend(loc="upper right")
    save(fig, name)


PC_AXES = [
    ("chunk_token_length", "Chunk\nlength", None),
    ("overlap_percentage", "Overlap", None),
    ("embedder", "Embedder", EMBEDDER_SHORT),
    ("retriever", "Retriever",
     {"base": "base", "ensemble": "ensemble", "bm25_only": "bm25", "mmr": "mmr"}),
    ("gen_model", "Gen.\nmodel", GEN_MODEL_SHORT),
    ("number of documents", "Num.\ndocs", None),
]


def parallel_coordinates(df, inc, name, color=("accuracy", "Accuracy"),
                         last_axis=("bert_f1_gold", "BERT F1", None)):
    """All configurations across hyperparameters and objectives.

    One objective colors the lines (``color``) while the other is the last
    axis (``last_axis``). Every line is drawn with the same style; a slight
    transparency keeps overlapping lines readable.
    """
    color_col, color_label = color
    axes = PC_AXES + [last_axis]
    data = df.dropna(subset=["bert_f1_gold"])
    front = inc.dropna(subset=["bert_f1_gold"])

    def normalize(frame, col, categories):
        if categories is not None:
            keys = list(categories)
            return frame[col].map({k: i / (len(keys) - 1)
                                   for i, k in enumerate(keys)})
        lo = min(data[col].min(), front[col].min())
        hi = max(data[col].max(), front[col].max())
        return (frame[col] - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(13, 6))
    xs = list(range(len(axes)))

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(data[color_col].min(), front[color_col].min()),
                         max(data[color_col].max(), front[color_col].max()))

    coords = [normalize(data, col, cats) for col, _, cats in axes]
    for _, row in data.iterrows():
        ys = [c.loc[row.name] for c in coords]
        ax.plot(xs, ys, color=cmap(norm(row[color_col])), alpha=0.6,
                linewidth=1.2, zorder=1)

    for x, (col, label, cats) in zip(xs, axes):
        ax.axvline(x, color="black", linewidth=1, zorder=3)
        if cats is not None:
            for i, short in enumerate(cats.values()):
                ax.text(x + 0.05, i / (len(cats) - 1), short, fontsize=14,
                        va="center", zorder=4,
                        bbox=dict(facecolor="white", alpha=0.85, pad=1,
                                  edgecolor="lightgray"))
        else:
            lo = min(data[col].min(), front[col].min())
            hi = max(data[col].max(), front[col].max())
            ax.text(x, -0.04, f"{lo:.2g}", fontsize=14, ha="center",
                    va="top", zorder=4)
            ax.text(x, 1.04, f"{hi:.2g}", fontsize=14, ha="center",
                    va="bottom", zorder=4)
    ax.set_xticks(xs, [label for _, label, _ in axes], fontsize=18)
    ax.tick_params(axis="x", pad=18)
    ax.set_yticks([])
    ax.set_xlim(-0.3, len(axes) - 0.7)
    ax.set_ylim(-0.1, 1.1)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label=color_label, pad=0.02)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, name)


def optimization_progress(df, name):
    """Best objective value reached so far, in file (evaluation) order."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    trials = range(1, len(df) + 1)
    for col, label, color in (
        ("1 - accuracy", "1 - Accuracy", "tab:blue"),
        ("1 - bert_f1_gold", "1 - BERTScore F1", "tab:green"),
    ):
        ax.scatter(trials, df[col], color=color, alpha=0.3, s=15)
        ax.step(trials, df[col].cummin().ffill(), where="post", color=color,
                linewidth=2, label=f"best {label}")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Objective value")
    ax.legend()
    save(fig, name)


def accuracy_vs_bert_by_genmodel(df, name):
    """Trade-off between the two quality objectives, by generation model."""
    data = df.dropna(subset=["bert_f1_gold"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=data, x="accuracy", y="bert_f1_gold", hue="gen_model_short",
        hue_order=list(GEN_MODEL_SHORT.values()), style="retriever",
        s=70, ax=ax,
    )
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("BERTScore F1")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=12)
    save(fig, name)


def correlation_accuracy_vs_bert(df, name, name_by_model):
    """Accuracy vs. BERTScore F1, as two standalone half-column figures:
    overall regression with correlation coefficients, and per-generation-model
    regressions. Fonts are enlarged so they stay readable at half-column."""
    from scipy import stats

    label_size, tick_size = 24, 20
    data = df.dropna(subset=["accuracy", "bert_f1_gold"])
    r, _ = stats.pearsonr(data["accuracy"], data["bert_f1_gold"])
    rho, _ = stats.spearmanr(data["accuracy"], data["bert_f1_gold"])

    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.regplot(data=data, x="accuracy", y="bert_f1_gold",
                scatter_kws=dict(s=40, alpha=0.6, color="tab:blue"),
                line_kws=dict(color="tab:red", linewidth=1.5), ax=ax)
    ax.annotate(
        f"Pearson $r$ = {r:.2f}\nSpearman $\\rho$ = {rho:.2f}"
        f"\n$n$ = {len(data)}",
        xy=(0.96, 0.05), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=tick_size,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    ax.set_xlabel("Accuracy", fontsize=label_size)
    ax.set_ylabel("BERTScore F1", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    save(fig, name)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    models = [m for m in GEN_MODEL_SHORT.values()
              if (data["gen_model_short"] == m).sum() >= 2]
    palette = dict(zip(models, sns.color_palette("tab10", len(models))))
    for model in models:
        sub = data[data["gen_model_short"] == model]
        rm, _ = stats.pearsonr(sub["accuracy"], sub["bert_f1_gold"])
        sns.regplot(data=sub, x="accuracy", y="bert_f1_gold", ci=None,
                    scatter_kws=dict(s=40, alpha=0.75),
                    line_kws=dict(linewidth=1.4),
                    color=palette[model], ax=ax,
                    label=f"{model} ($r$={rm:.2f})")
    ax.legend(fontsize=15, loc="lower right", labelspacing=0.25,
              handletextpad=0.4, borderaxespad=0.2, borderpad=0.3)
    ax.set_xlabel("Accuracy", fontsize=label_size)
    ax.set_ylabel("BERTScore F1", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    save(fig, name_by_model)


def incumbent_rows(inc):
    rows = []
    for _, r in inc.iterrows():
        retriever = r["retriever"]
        if pd.notna(r["mmr_fetch_k"]):
            retriever += (f" ({int(r['mmr_fetch_k'])}, "
                          f"{r['mmr_lambda_mult']:.2f})")
        bert = f"{r['bert_f1_gold']:.3f}" if pd.notna(r["bert_f1_gold"]) else "—"
        rows.append([
            r["id"], EMBEDDER_SHORT[r["embedder"]], retriever,
            GEN_MODEL_SHORT[r["gen_model"]], int(r["chunk_token_length"]),
            f"{r['overlap_percentage']:.2f}", int(r["number of documents"]),
            f"{r['accuracy']:.2f}", bert,
        ])
    return rows


TABLE_HEADER = ["ID", "Embedder", "Retriever", "Gen. model", "Chunk",
                "Overlap", "Docs", "Accuracy", "BERT F1"]


def incumbents_table_pdf(inc, name):
    rows = incumbent_rows(inc)
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(rows) + 1.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=TABLE_HEADER, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    table.auto_set_column_width(range(len(TABLE_HEADER)))
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#3a5e8c")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eef2f7")
        cell.set_edgecolor("#cccccc")
    save(fig, name)


def incumbents_table_tex(inc, name):
    rows = incumbent_rows(inc)
    lines = [
        "% Auto-generated by generate_figures.py — requires \\usepackage{booktabs}",
        "\\begin{table}",
        "    \\centering",
        "    \\caption{Pareto-front incumbents found by SMAC3, sorted by accuracy.",
        "    %",
        "        For the \\emph{mmr} retriever, fetch-$k$ and $\\lambda$ are reported in parentheses next to the retriever name.",
        "    %",
        "        The BERTScore of I4 is missing because its evaluation crashed.}",
        "    \\label{tab:incumbents}",
        "    \\resizebox{\\linewidth}{!}{%",
        "    \\begin{tabular}{llllrrrrr}",
        "        \\toprule",
        "        " + " & ".join(TABLE_HEADER) + " \\\\",
        "        \\midrule",
    ]
    for r in rows:
        cells = [str(c).replace("—", "---").replace("_", "\\_") for c in r]
        lines.append("        " + " & ".join(cells) + " \\\\")
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}}",
        "\\end{table}",
        "",
    ]
    (FIGURES / name).write_text("\n".join(lines))
    print(f"  wrote figures/{name}")


def heatmap(df, index, columns, values, fname, fmt=".2f"):
    pivot = df.pivot_table(index=index, columns=columns, values=values,
                           aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="viridis", ax=ax,
                annot_kws={"size": 15})
    # combinations that were never evaluated show as gray cells
    ax.set_facecolor("0.85")
    save(fig, fname)


def kde_corner(df, name):
    """All pairwise KDEs of the numeric hyperparameters in one corner plot."""
    cols = {
        "number of documents": "Num. docs",
        "chunk_token_length": "Chunk length",
        "overlap_percentage": "Overlap",
    }
    data = df[list(cols)].rename(columns=cols)
    grid = sns.pairplot(data, kind="kde", corner=True, height=3.2,
                        plot_kws=dict(fill=True), diag_kws=dict(fill=True))
    grid.figure.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(grid.figure)
    print(f"  wrote figures/{name}")


def kde_objective_space(df, inc, name):
    """KDE of the objective space with marginals; incumbents overlaid."""
    data = df.dropna(subset=["1 - accuracy", "number of documents"])
    grid = sns.jointplot(
        data=data, x="1 - accuracy", y="number of documents",
        kind="kde", fill=True, height=6,
    )
    grid.ax_joint.scatter(data["1 - accuracy"], data["number of documents"],
                          marker="x", color="tab:blue", alpha=0.6, s=25)
    grid.ax_joint.scatter(inc["1 - accuracy"], inc["number of documents"],
                          marker="x", color="red", s=70, linewidths=2,
                          label="Incumbents")
    grid.ax_joint.legend(loc="upper right")
    grid.set_axis_labels("1 - Accuracy", "Number of Documents")
    grid.figure.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(grid.figure)
    print(f"  wrote figures/{name}")


def kde_top_configurations(df, name):
    """Where the explored vs. the best-performing configurations live."""
    threshold = df["accuracy"].quantile(0.75)
    top = df[df["accuracy"] >= threshold]
    pairs = [
        ("number of documents", "chunk_token_length",
         "Number of Documents", "Chunk Length"),
        ("chunk_token_length", "overlap_percentage",
         "Chunk Length", "Overlap Percentage"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (x, y, xlabel, ylabel) in zip(axes, pairs):
        sns.kdeplot(data=df, x=x, y=y, fill=True, levels=8, ax=ax)
        sns.kdeplot(data=top, x=x, y=y, levels=5, color="crimson",
                    linewidths=1.5, ax=ax)
        ax.scatter(top[x], top[y], color="crimson", s=18, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    save(fig, name)


def kde_accuracy_by_retriever(df, name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.kdeplot(data=df, x="accuracy", hue="retriever", fill=True,
                alpha=0.25, common_norm=False, clip=(0, 1), ax=ax)
    ax.set_xlabel("Accuracy")
    save(fig, name)


def kde_ridgeline(df, value, value_label, name, clip=None):
    """Ridgeline of per-generation-model KDEs."""
    import numpy as np
    from scipy.stats import gaussian_kde

    data = df.dropna(subset=[value])
    order = [m for m in GEN_MODEL_SHORT.values()
             if (data["gen_model_short"] == m).sum() >= 3]
    lo, hi = data[value].min(), data[value].max()
    pad = 0.05 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 300)
    if clip is not None:
        xs = np.clip(xs, *clip)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, model in enumerate(reversed(order)):
        vals = data.loc[data["gen_model_short"] == model, value]
        ys = gaussian_kde(vals)(xs)
        ys = ys / ys.max() * 0.85
        color = cmap(i / max(len(order) - 1, 1))
        ax.fill_between(xs, i, i + ys, color=color, alpha=0.7, zorder=2)
        ax.plot(xs, i + ys, color="black", linewidth=0.8, zorder=3)
        ax.scatter(vals, [i] * len(vals), marker="|", color="black", s=40,
                   zorder=4)
    ax.set_yticks(range(len(order)), list(reversed(order)))
    ax.set_xlabel(value_label)
    save(fig, name)


def main():
    df, inc = load_data()
    print(f"{len(df)} evaluated configurations, {len(inc)} incumbents")

    docs_order = list(range(1, 21))
    gen_order = list(GEN_MODEL_SHORT.values())
    emb_order = list(EMBEDDER_SHORT.values())

    # --- Pareto front scatter plots (objective space) ---
    pareto_scatter(
        df, inc, "1 - accuracy", "number of documents",
        "1 - Accuracy", "Number of Documents", "result.pdf",
    )
    pareto_scatter(
        df, inc, "1 - bert_f1_gold", "number of documents",
        "1 - BERTScore F1", "Number of Documents",
        "result-bert.pdf",
    )
    pareto_scatter(
        df, inc, "1 - accuracy", "1 - bert_f1_gold",
        "1 - Accuracy", "1 - BERTScore F1",
        "result-accuracy-vs-bert.pdf",
    )

    # --- Accuracy boxplots ---
    boxplot(
        df, "number of documents", "accuracy", "", "Accuracy",
        "boxplot-accuracy-number-of-documents.pdf", order=docs_order,
        rotation=0, tick_every=5,
    )
    boxplot(
        df, "embedder_short", "accuracy", "Embedder", "Accuracy",
        "boxplot-accuracy-embedder-and-retriever.pdf",
        hue="retriever", rotation=30, order=emb_order,
        figsize=(9, 4.5), legend_top=True,
    )
    boxplot(
        df, "gen_model_short", "accuracy", "Generation Model", "Accuracy",
        "boxplot-accuracy-gen-model.pdf", rotation=30, order=gen_order,
    )
    paired_boxplots(
        df, "accuracy", "Accuracy",
        "boxplot-accuracy-number-of-documents-and-embedder.pdf",
        docs_order, emb_order,
    )
    binned_boxplot(
        df, "chunk_token_length", "accuracy", 6,
        "Chunk Length Bin", "Accuracy",
        "boxplot-accuracy-chunk-length.pdf",
    )
    binned_boxplot(
        df, "overlap_percentage", "accuracy", 6,
        "Overlap Percentage Bin", "Accuracy",
        "boxplot-accuracy-overlap-percentage.pdf",
    )
    paired_binned_boxplots(
        df, "accuracy", "Accuracy",
        "boxplot-accuracy-chunk-length-and-overlap.pdf",
    )

    # --- BERTScore boxplots (generation quality vs. gold answer) ---
    bert = df.dropna(subset=["bert_f1_gold"])
    boxplot(
        bert, "number of documents", "bert_f1_gold", "",
        "BERTScore F1",
        "boxplot-bert-number-of-documents.pdf", order=docs_order,
        rotation=0, tick_every=5,
    )
    boxplot(
        bert, "embedder_short", "bert_f1_gold", "Embedder",
        "BERTScore F1",
        "boxplot-bert-embedder-and-retriever.pdf",
        hue="retriever", rotation=30, order=emb_order,
        figsize=(9, 4.5), legend_top=True,
    )
    boxplot(
        bert, "gen_model_short", "bert_f1_gold", "Generation Model",
        "BERTScore F1",
        "boxplot-bert-gen-model.pdf", rotation=30, order=gen_order,
    )
    paired_boxplots(
        bert, "bert_f1_gold", "BERTScore F1",
        "boxplot-bert-number-of-documents-and-embedder.pdf",
        docs_order, emb_order,
    )
    binned_boxplot(
        bert, "chunk_token_length", "bert_f1_gold", 6,
        "Chunk Length Bin",
        "BERTScore F1", "boxplot-bert-chunk-length.pdf",
    )
    binned_boxplot(
        bert, "overlap_percentage", "bert_f1_gold", 6,
        "Overlap Percentage Bin",
        "BERTScore F1", "boxplot-bert-overlap-percentage.pdf",
    )
    paired_binned_boxplots(
        bert, "bert_f1_gold", "BERTScore F1",
        "boxplot-bert-chunk-length-and-overlap.pdf",
    )

    # --- Insightful renderings: 3D objective space and aggregate views ---
    scatter_3d(df, inc, "result-3d.pdf")
    scatter_3d_interactive(df, inc, "result-3d.html")
    pareto_surface_3d(df, inc, "result-3d-surface.pdf")
    pareto_surface_3d_interactive(df, inc, "result-3d-surface.html")
    incumbents_table_pdf(inc, "incumbents-table.pdf")
    incumbents_table_tex(inc, "incumbents-table.tex")
    pareto_colored_by_bert(df, inc, "result-colored-by-bert.pdf")
    parallel_coordinates(df, inc, "parallel-coordinates.pdf")
    parallel_coordinates(
        df, inc, "parallel-coordinates-colored-by-bert.pdf",
        color=("bert_f1_gold", "BERTScore F1"),
        last_axis=("accuracy", "Accuracy", None),
    )
    optimization_progress(df, "optimization-progress.pdf")
    accuracy_vs_bert_by_genmodel(df, "scatter-accuracy-vs-bert-by-gen-model.pdf")
    correlation_accuracy_vs_bert(
        df, "correlation-accuracy-vs-bert-f1.pdf",
        "correlation-accuracy-vs-bert-f1-by-gen-model.pdf",
    )
    heatmap(
        df, "embedder_short", "retriever", "accuracy",
        "heatmap-accuracy-embedder-retriever.pdf",
    )
    heatmap(
        df, "gen_model_short", "retriever", "bert_f1_gold",
        "heatmap-bert-gen-model-retriever.pdf", fmt=".3f",
    )

    # --- KDE plots of the explored configuration space ---
    kde(
        df, "number of documents", "chunk_token_length",
        "Number of Documents", "Chunk Length",
        "kde-number-of-documents-and-chunk-length.pdf",
    )
    kde(
        df, "number of documents", "overlap_percentage",
        "Number of Documents", "Overlap Percentage",
        "kde-number-of-documents-and-overlap-percentage.pdf",
    )
    kde(
        df, "chunk_token_length", "overlap_percentage",
        "Chunk Length", "Overlap Percentage",
        "kde-chunk-length-and-overlap-percentage.pdf",
    )
    mmr = df[df["retriever"] == "mmr"]
    kde(
        mmr, "mmr_fetch_k", "mmr_lambda_mult",
        "MMR fetch-k", "MMR lambda",
        "kde-mmr-fetch-k-and-lambda.pdf",
    )
    kde_corner(df, "kde-pairs.pdf")
    kde_objective_space(df, inc, "kde-objective-space.pdf")
    kde_top_configurations(df, "kde-top-configurations.pdf")
    kde_accuracy_by_retriever(df, "kde-accuracy-by-retriever.pdf")
    kde_ridgeline(df, "accuracy", "Accuracy",
                  "kde-ridgeline-accuracy-by-gen-model.pdf", clip=(0, 1))
    kde_ridgeline(df, "bert_f1_gold", "BERTScore F1",
                  "kde-ridgeline-bert-by-gen-model.pdf")


if __name__ == "__main__":
    main()
