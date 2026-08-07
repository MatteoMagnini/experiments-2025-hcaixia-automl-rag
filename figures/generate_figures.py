"""Figures for the AutoML (SMAC3) half of the paper.

Input
-----
``results/incumbents.csv_results.csv``
    One row per configuration SMAC3 evaluated.  ``status == "failed"`` marks a
    trial that crashed before producing any metric; a missing ``bert_f1_gold``
    marks one whose retrieval succeeded but whose generative evaluation did not.
``results/incumbents.csv``
    The Pareto-front incumbents.  A crashed generative evaluation is stored as
    the placeholder loss ``1 - bert_f1_gold == 1.0`` (SMAC cannot take NaN
    costs), which is restored to NA on load so it is never plotted as a score.

Output
------
``figures/figures/`` — one vector PDF per figure, drawn at its final printed
size in the shared style of ``figures/paper_style.py``.

Two conventions hold across this file and its companion under
``poe-retrieval-experiment/figures``, so a component keeps one identity
throughout the paper:

*One name, one colour, one position.*  ``RETRIEVER_COLOR`` pins ``ensemble`` to
blue and ``mmr`` to orange, matching the retrieval figures where the same two
strategies reappear; ``EMBEDDER_SHORT`` and ``GEN_MODEL_SHORT`` fix both the
short names and the order they are listed in, which is the order used in the
configuration-space table of the paper.

*Figures are drawn at their final printed size.*  Every size comes from
``ps.WIDE``/``ps.COLUMN``, so a figure included at ``width=\\linewidth`` is not
rescaled and its labels land at body-text size.  This is why nothing here sets
a font size by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "results"
FIGURES = ROOT / "figures"

# every path is derived from this file's location, so the script runs the same
# from any working directory
sys.path.insert(0, str(ROOT))
import paper_style as ps  # noqa: E402

RESULTS_FILE = DATA / "incumbents.csv_results.csv"
INCUMBENTS_FILE = DATA / "incumbents.csv"

# ---------------------------------------------------------------------------
# Naming, ordering and colour
# ---------------------------------------------------------------------------

#: Embedders, in the order the configuration-space table of the paper lists
#: them.  Every categorical axis showing embedders uses this order.
EMBEDDER_SHORT = {
    "bert-base-italian-xxl-cased": "bert-xxl (it)",
    "granite-embedding-107m": "granite-107m",
    "granite-embedding-278m": "granite-278m",
    "nomic-embed-text-v2-moe": "nomic-v2-moe",
    "qwen3-embedding-0.6b": "qwen3-0.6b",
    "qwen3-embedding-4b": "qwen3-4b",
}

#: Generation models, ordered by family and then by scale.
GEN_MODEL_SHORT = {
    "google/gemma-3-4b-it": "gemma-3-4b",
    "google/gemma-3-12b-it": "gemma-3-12b",
    "google/gemma-3-27b-it": "gemma-3-27b",
    "qwen/qwen3-8b": "qwen3-8b",
    "qwen/qwen3-14b": "qwen3-14b",
    "qwen/qwen3-32b": "qwen3-32b",
}

RETRIEVER_ORDER = ("base", "bm25_only", "ensemble", "mmr")

#: Colour follows the retriever, not its rank.  ``ensemble`` and ``mmr`` keep
#: the hues they carry in the downstream retrieval figures, so a reader who has
#: learnt "blue is ensemble, orange is MMR" there reads these the same way.
RETRIEVER_COLOR = {
    "ensemble": ps.SERIES["blue"],
    "mmr": ps.SERIES["orange"],
    "base": ps.SERIES["yellow"],
    "bm25_only": ps.SERIES["magenta"],
}

# Semantic roles for the objective-space figures, mapped onto the shared
# palette once so the code below says what a colour *means*.
EVALUATED = ps.SERIES["blue"]   # a configuration SMAC actually evaluated
INCUMBENT = ps.SERIES["red"]    # a point on the Pareto front
NEUTRAL = "0.72"                # no categorical meaning attached

ACCURACY_LABEL = "Accuracy"
BERT_LABEL = "BERTScore F1"
DOCS_LABEL = "Number of documents"

#: Objectives as SMAC minimises them, with the labels used on every axis.
OBJECTIVES = ["1 - accuracy", "number of documents", "1 - bert_f1_gold"]
OBJECTIVE_LABELS = ["1 - Accuracy", DOCS_LABEL, f"1 - {BERT_LABEL}"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _require(frame: pd.DataFrame, columns, source: Path) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{source.name} is missing column(s): {', '.join(missing)}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run the SMAC experiment first "
            "(python -m experiments) so the results are written to results/."
        )
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty")
    return frame


def _short(frame: pd.DataFrame, column: str, mapping: dict[str, str]) -> pd.Series:
    """Map a categorical column to its short names, refusing silent NaNs.

    An unmapped value would otherwise disappear from every ordered plot — the
    category simply would not be drawn — so it is worth an explicit failure.
    """
    unknown = sorted(set(frame[column].dropna()) - set(mapping))
    if unknown:
        raise KeyError(f"unknown {column} value(s): {', '.join(unknown)}; "
                       f"add them to the short-name table in {Path(__file__).name}")
    return frame[column].map(mapping)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The evaluated configurations and the Pareto-front incumbents.

    Trials that crashed outright are dropped.  Trials whose generative
    evaluation failed are kept with ``bert_f1_gold`` as NA: their retrieval
    accuracy is real data and belongs in the accuracy figures, while every
    BERTScore figure drops them explicitly.
    """
    df = _read_csv(RESULTS_FILE)
    _require(df, ["status", "1 - accuracy", "number of documents", "bert_f1_gold",
                  "embedder", "retriever", "gen_model", "chunk_token_length",
                  "overlap_percentage"], RESULTS_FILE)
    evaluated = len(df)
    df = df[df["status"] != "failed"].copy()
    df["accuracy"] = 1 - df["1 - accuracy"]
    df["number of documents"] = df["number of documents"].astype(int)
    df["embedder_short"] = _short(df, "embedder", EMBEDDER_SHORT)
    df["gen_model_short"] = _short(df, "gen_model", GEN_MODEL_SHORT)

    inc = _read_csv(INCUMBENTS_FILE)
    _require(inc, ["1 - accuracy", "number of documents", "1 - bert_f1_gold",
                   "embedder", "retriever", "gen_model"], INCUMBENTS_FILE)
    # a crashed generative evaluation is recorded as the worst possible loss
    inc.loc[inc["1 - bert_f1_gold"] >= 1.0, "1 - bert_f1_gold"] = pd.NA
    inc["accuracy"] = 1 - inc["1 - accuracy"]
    inc["bert_f1_gold"] = 1 - inc["1 - bert_f1_gold"]
    inc["number of documents"] = inc["number of documents"].astype(int)
    # by retrieval depth: it is the objective the front is read along, and the
    # ordering the accuracy discussion in the paper follows
    inc = inc.sort_values("number of documents").reset_index(drop=True)
    inc["id"] = [f"I{i + 1}" for i in range(len(inc))]

    print(f"{evaluated} configurations evaluated, {len(df)} completed retrieval, "
          f"{int(df['bert_f1_gold'].notna().sum())} also scored by BERTScore; "
          f"{len(inc)} incumbents "
          f"({int(inc['bert_f1_gold'].notna().sum())} with a BERTScore)")
    return df, inc


def save(fig, name: str):
    return ps.save(fig, FIGURES, name)


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def retriever_palette() -> dict[str, str]:
    return {r: RETRIEVER_COLOR[r] for r in RETRIEVER_ORDER}


def bin_labels(values: pd.Series, bins: int, fmt: str) -> tuple[pd.Series, list[str]]:
    """Equal-width bins rendered as ``low-high`` labels, in order.

    Interval notation is dropped: at tick-label size the brackets cost width
    without telling the reader anything the ordering does not already say.
    """
    cut = pd.cut(values, bins=bins)
    labels = {c: f"{fmt.format(c.left)}\N{EN DASH}{fmt.format(c.right)}"
              for c in cut.cat.categories}
    return cut.map(labels), list(labels.values())


def rotate_ticks(ax, angle: int = 30) -> None:
    """Rotate the x tick labels and right-align them under their tick."""
    ax.tick_params(axis="x", rotation=angle)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def sparse_ticks(ax, every: int) -> None:
    """Keep one x tick label in ``every``; with 20 categories on a 5.5in axis
    the full set collides whatever the point size."""
    for i, label in enumerate(ax.get_xticklabels()):
        if i != 0 and (i + 1) % every:
            label.set_visible(False)


def style_boxes(ax) -> None:
    """Hairline box outlines and median lines, in text ink.

    seaborn draws boxes with a coloured edge of the same hue as the fill, which
    at this size reads as a slightly darker fill rather than as an outline.
    """
    for patch in ax.patches:
        patch.set_edgecolor(ps.INK["secondary"])
        patch.set_linewidth(0.6)
    for line in ax.lines:
        line.set_linewidth(0.7)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_pareto_front(df: pd.DataFrame, inc: pd.DataFrame, name: str) -> None:
    """The objective space SMAC3 searched, with the Pareto front marked.

    Two of the three objectives are the axes and the third is the colour, so
    the front can be read as a front: points that look dominated in the plane
    are the ones that pay for it on the colour axis.
    """
    data = df.dropna(subset=OBJECTIVES)
    front = inc.dropna(subset=OBJECTIVES)

    fig, ax = plt.subplots(figsize=(ps.WIDE, ps.WIDE * 0.46))
    dots = ax.scatter(data["1 - accuracy"], data["number of documents"],
                      c=data["bert_f1_gold"], cmap=ps.SEQUENTIAL, s=26,
                      edgecolors=ps.INK["surface"], linewidths=0.4, zorder=2)
    ax.scatter(front["1 - accuracy"], front["number of documents"],
               facecolors="none", edgecolors=INCUMBENT, s=95, linewidths=1.1,
               label="Incumbents", zorder=3)
    bar = fig.colorbar(dots, ax=ax, pad=0.02)
    bar.set_label(BERT_LABEL, color=ps.INK["secondary"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2.5, color=ps.INK["axis"],
                       labelcolor=ps.INK["muted"])

    ax.set_xlabel(OBJECTIVE_LABELS[0])
    ax.set_ylabel(DOCS_LABEL)
    ax.legend(loc="upper right")
    ps.grid_axis(ax, "both")
    fig.tight_layout()
    save(fig, name)


def fig_accuracy_drivers(df: pd.DataFrame, name: str) -> None:
    """Retrieval depth and the embedder/retriever pair, against accuracy.

    Stacked rather than side by side: the lower panel alone needs 24 box slots,
    and at the 5.5in text width of the venue two such panels leave neither one
    enough room for its category labels.
    """
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(ps.WIDE, 3.6), sharey=True,
        gridspec_kw=dict(height_ratios=[1, 1.1]))

    # neutral grey: the retriever palette of the lower panel does not apply here
    sns.boxplot(data=df, x="number of documents", y="accuracy",
                order=list(range(1, 21)), color=NEUTRAL, width=0.7,
                fliersize=2.0, ax=top)
    top.set_xlabel(DOCS_LABEL)
    top.set_ylabel(ACCURACY_LABEL)
    sparse_ticks(top, 5)
    style_boxes(top)
    ps.grid_axis(top, "y")
    ps.panel_title(top, "a", "Retrieval depth")

    sns.boxplot(data=df, x="embedder_short", y="accuracy", hue="retriever",
                order=list(EMBEDDER_SHORT.values()), hue_order=RETRIEVER_ORDER,
                palette=retriever_palette(), width=0.75, fliersize=2.0,
                ax=bottom)
    bottom.set_xlabel("Embedder")
    bottom.set_ylabel(ACCURACY_LABEL)
    rotate_ticks(bottom, 18)
    style_boxes(bottom)
    ps.grid_axis(bottom, "y")
    ps.panel_title(bottom, "b", "Embedder and retriever")

    # the retriever legend belongs to the figure, not to one panel: placed
    # inside panel (b) it would sit on top of its title or its boxes
    handles, labels = bottom.get_legend_handles_labels()
    bottom.legend_.remove()
    fig.tight_layout(h_pad=1.6)
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    save(fig, name)


def fig_chunking(df: pd.DataFrame, name: str) -> None:
    """Accuracy against the two chunking hyperparameters, binned.

    Both panels hold six bins, so they do fit side by side at the text width.
    """
    fig, axes = plt.subplots(1, 2, figsize=(ps.WIDE, 2.15), sharey=True,
                             layout="constrained")
    panels = [("chunk_token_length", "Chunk length (tokens)", "{:.0f}", "a"),
              ("overlap_percentage", "Overlap percentage", "{:.2f}", "b")]
    for ax, (column, xlabel, fmt, letter) in zip(axes, panels):
        binned = df.copy()
        binned["bin"], order = bin_labels(binned[column], 6, fmt)
        sns.boxplot(data=binned, x="bin", y="accuracy", order=order,
                    color=NEUTRAL, width=0.7, fliersize=2.0, ax=ax)
        ax.set_xlabel(xlabel)
        rotate_ticks(ax, 30)
        style_boxes(ax)
        ps.grid_axis(ax, "y")
        ps.panel_title(ax, letter)
    axes[0].set_ylabel(ACCURACY_LABEL)
    axes[1].set_ylabel("")
    save(fig, name)


PC_AXES = [
    ("chunk_token_length", "Chunk\nlength", None),
    ("overlap_percentage", "Overlap", None),
    ("embedder", "Embedder", EMBEDDER_SHORT),
    ("retriever", "Retriever", {r: r.replace("_only", "") for r in RETRIEVER_ORDER}),
    ("gen_model", "Gen.\nmodel", GEN_MODEL_SHORT),
    ("number of documents", "Num.\ndocs", None),
    ("bert_f1_gold", "BERT\nF1", None),
]


def fig_parallel_coordinates(df: pd.DataFrame, name: str) -> None:
    """Every evaluated configuration across all hyperparameters at once.

    One objective is the colour and the other the last axis, so a line can be
    followed from its settings to both of its quality scores.  Lines are drawn
    with one style throughout — the colour already carries the only ranking
    there is, and a second encoding would imply a second variable.
    """
    data = df.dropna(subset=["bert_f1_gold"])
    if data.empty:
        raise ValueError("no configuration has a BERTScore; nothing to draw")

    def normalise(column: str, categories) -> pd.Series:
        if categories is not None:
            keys = list(categories)
            return data[column].map({k: i / (len(keys) - 1)
                                     for i, k in enumerate(keys)})
        lo, hi = data[column].min(), data[column].max()
        return (data[column] - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(ps.WIDE, 3.0))
    xs = list(range(len(PC_AXES)))
    norm = plt.Normalize(data["accuracy"].min(), data["accuracy"].max())
    coords = [normalise(column, categories) for column, _, categories in PC_AXES]

    for index, row in data.iterrows():
        ax.plot(xs, [c.loc[index] for c in coords],
                color=ps.SEQUENTIAL(norm(row["accuracy"])), alpha=0.55,
                linewidth=0.7, zorder=1)

    # the category names have to fit in the gap between two axes, which at the
    # text width of the venue is about 45pt — hence the reduced size here, the
    # only place in this file where a size is not taken straight from the style
    label_size = plt.rcParams["xtick.labelsize"] * 0.78
    for x, (column, _, categories) in zip(xs, PC_AXES):
        ax.axvline(x, color=ps.INK["axis"], linewidth=0.8, zorder=3)
        if categories is not None:
            for i, short in enumerate(categories.values()):
                ax.text(x + 0.05, i / (len(categories) - 1), short, zorder=4,
                        va="center", fontsize=label_size,
                        color=ps.INK["secondary"],
                        bbox=dict(facecolor=ps.INK["surface"], alpha=0.85,
                                  pad=0.8, edgecolor="none"))
        else:
            lo, hi = data[column].min(), data[column].max()
            for value, y, va in ((lo, -0.02, "top"), (hi, 1.02, "bottom")):
                ax.text(x, y, f"{value:.3g}", ha="center", va=va, zorder=4,
                        fontsize=label_size, color=ps.INK["muted"])

    ax.set_xticks(xs, [label for _, label, _ in PC_AXES])
    ax.tick_params(axis="x", pad=10, length=0, labelcolor=ps.INK["secondary"],
                   labelsize=plt.rcParams["xtick.labelsize"] * 0.9)
    ax.set_yticks([])
    ax.set_xlim(-0.3, len(PC_AXES) - 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ps.SEQUENTIAL),
                       ax=ax, pad=0.01, fraction=0.035)
    bar.set_label(ACCURACY_LABEL, color=ps.INK["secondary"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2.5, color=ps.INK["axis"],
                       labelcolor=ps.INK["muted"])
    fig.tight_layout()
    save(fig, name)


def fig_correlation(df: pd.DataFrame, name: str) -> None:
    """The two quality objectives against each other, pooled and per model.

    Both panels in one figure rather than two half-width includes: the panels
    are then guaranteed the same height and type size, and the figure carries
    a single caption for a single comparison.
    """
    data = df.dropna(subset=["accuracy", "bert_f1_gold"])
    r, _ = stats.pearsonr(data["accuracy"], data["bert_f1_gold"])
    rho, _ = stats.spearmanr(data["accuracy"], data["bert_f1_gold"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(ps.WIDE, 2.5), sharey=True,
                                      layout="constrained")

    sns.regplot(data=data, x="accuracy", y="bert_f1_gold",
                scatter_kws=dict(s=16, alpha=0.65, color=EVALUATED),
                line_kws=dict(color=INCUMBENT, linewidth=1.2), ax=left)
    left.annotate(f"Pearson $r$ = {r:.2f}\nSpearman $\\rho$ = {rho:.2f}\n"
                  f"$n$ = {len(data)}",
                  xy=(0.04, 0.96), xycoords="axes fraction", ha="left",
                  va="top", color=ps.INK["secondary"],
                  fontsize=plt.rcParams["xtick.labelsize"])
    left.set_ylabel(BERT_LABEL)
    ps.panel_title(left, "a", "All configurations")

    models = [m for m in GEN_MODEL_SHORT.values()
              if (data["gen_model_short"] == m).sum() >= 3]
    palette = dict(zip(models, ps.cycle(len(models))))
    for model in models:
        subset = data[data["gen_model_short"] == model]
        r_model, _ = stats.pearsonr(subset["accuracy"], subset["bert_f1_gold"])
        sns.regplot(data=subset, x="accuracy", y="bert_f1_gold", ci=None,
                    scatter_kws=dict(s=14, alpha=0.8),
                    line_kws=dict(linewidth=1.0), color=palette[model],
                    ax=right, label=f"{model} ($r$={r_model:.2f})")
    right.set_ylabel("")
    ps.panel_title(right, "b", "By generation model")

    for ax in (left, right):
        ax.set_xlabel(ACCURACY_LABEL)
        ps.grid_axis(ax, "both")

    # six model entries do not fit inside a half-width panel without covering
    # the fits they label, so they go under the figure
    handles, labels = right.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, columnspacing=1.0)
    save(fig, name)


KDE_COLUMNS = {
    "number of documents": "Num. docs",
    "chunk_token_length": "Chunk length",
    "overlap_percentage": "Overlap",
}


def fig_search_density(df: pd.DataFrame, name: str) -> None:
    """Where in the numeric part of the space SMAC3 actually spent its budget.

    A corner plot: the diagonal is where each hyperparameter was sampled on its
    own, the lower triangle where pairs of them were sampled together.  Built
    from plain subplots rather than a seaborn ``PairGrid`` so the panel grid is
    laid out at an exact printed size and the marginal densities keep their own
    vertical scale instead of inheriting the shared one.
    """
    columns = list(KDE_COLUMNS)
    n = len(columns)
    fig, axes = plt.subplots(n, n, figsize=(ps.WIDE, ps.WIDE * 0.62),
                             sharex="col")

    for row in range(n):
        for col in range(n):
            ax = axes[row][col]
            if col > row:
                ax.set_visible(False)
                continue
            if row == col:
                sns.kdeplot(x=df[columns[col]], fill=True, color=EVALUATED,
                            linewidth=0.9, ax=ax)
                # a density in absolute units means nothing next to the other
                # panels; the shape is the whole message
                ax.set_yticks([])
                ax.set_ylabel("")
                ps.despine(ax, left=True)
            else:
                sns.kdeplot(x=df[columns[col]], y=df[columns[row]], fill=True,
                            levels=7, color=EVALUATED, ax=ax)
            ps.grid_axis(ax, "both")
            if col == 0 and row != 0:
                ax.set_ylabel(KDE_COLUMNS[columns[row]])
            else:
                ax.set_ylabel("")
            if row == n - 1:
                ax.set_xlabel(KDE_COLUMNS[columns[col]])
            else:
                ax.set_xlabel("")
            if col != 0 or row == 0:
                ax.tick_params(labelleft=False)

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save(fig, name)


def fig_generation_model(df: pd.DataFrame, name: str) -> None:
    """Answer quality by generation model.

    Retrieval accuracy is not shown against the generation model on purpose:
    accuracy is computed from the retrieved chunks alone, so it cannot depend
    on which model wrote the answer, and any difference across models would be
    an artefact of which configurations SMAC happened to pair them with.
    """
    data = df.dropna(subset=["bert_f1_gold"])
    fig, ax = plt.subplots(figsize=(ps.WIDE, 2.2))
    sns.boxplot(data=data, x="gen_model_short", y="bert_f1_gold",
                order=list(GEN_MODEL_SHORT.values()), color=NEUTRAL,
                width=0.6, fliersize=2.0, ax=ax)
    ax.set_xlabel("Generation model")
    ax.set_ylabel(BERT_LABEL)
    style_boxes(ax)
    ps.grid_axis(ax, "y")
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------------------
# Incumbent table
# ---------------------------------------------------------------------------

TABLE_HEADER = ["Chunk", "Overlap", "Embedder", "Gen.\\ model", "Retriever",
                "\\#Docs", "Acc.", "BERTScore F1"]


def incumbents_table_tex(inc: pd.DataFrame, name: str) -> None:
    """The Pareto front as the LaTeX table the paper includes.

    Generated rather than hand-kept so the table cannot drift from the CSV.

    Incumbents whose generative evaluation crashed are left out: every column of
    the table except one would be real and the last would be a dash, which reads
    as a poor score rather than as a missing measurement.  The caption names them
    instead, and the count printed below says how many were dropped, so the
    omission stays visible.  This matches the figures, which drop the same rows.
    """
    complete = inc[inc["bert_f1_gold"].notna()]
    dropped = len(inc) - len(complete)

    rows = []
    for _, r in complete.iterrows():
        retriever = r["retriever"].replace("_", "\\_")
        if pd.notna(r.get("mmr_fetch_k")):
            retriever += (f" ($k={int(r['mmr_fetch_k'])}$, "
                          f"$\\lambda={r['mmr_lambda_mult']:.2f}$)")
        bert = f"{r['bert_f1_gold']:.2f}"
        rows.append(" & ".join([
            f"{int(r['chunk_token_length'])}",
            f"{r['overlap_percentage']:.2f}",
            EMBEDDER_SHORT[r["embedder"]],
            GEN_MODEL_SHORT[r["gen_model"]],
            retriever,
            f"{int(r['number of documents'])}",
            f"{r['accuracy']:.2f}",
            bert,
        ]) + " \\\\")

    lines = [
        f"% Auto-generated by figures/{Path(__file__).name} — do not edit by hand.",
        "% Requires \\usepackage{booktabs}.",
        "\\begin{tabular}{rrlllrrr}",
        "    \\toprule",
        "    " + " & ".join(f"\\textbf{{{h}}}" for h in TABLE_HEADER) + " \\\\",
        "    \\midrule",
        *[f"    {row}" for row in rows],
        "    \\bottomrule",
        "\\end{tabular}",
        "",
    ]
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / name).write_text("\n".join(lines))
    note = f" ({dropped} incumbent(s) left out, no BERTScore)" if dropped else ""
    print(f"  wrote {FIGURES.name}/{name}: {len(rows)} rows{note}")


# ---------------------------------------------------------------------------

def main() -> None:
    ps.use_paper_style()
    df, inc = load_data()

    fig_pareto_front(df, inc, "pareto-front.pdf")
    fig_accuracy_drivers(df, "accuracy-depth-embedder-retriever.pdf")
    fig_chunking(df, "accuracy-chunking.pdf")
    fig_parallel_coordinates(df, "parallel-coordinates.pdf")
    fig_correlation(df, "accuracy-vs-bert-f1.pdf")
    fig_search_density(df, "search-density.pdf")
    fig_generation_model(df, "bert-f1-by-generation-model.pdf")
    incumbents_table_tex(inc, "incumbents-table.tex")


if __name__ == "__main__":
    main()
