"""The single visual style shared by every figure in the paper.

Both figure sets — the AutoML/SMAC figures in this directory and the retrieval
figures under ``poe-retrieval-experiment/figures`` — import this module, so the
paper has one palette, one type scale and one set of chart conventions.

Two things it fixes relative to ad-hoc per-script styling:

*Colour is chosen, not inherited.*  The categorical palette below is a
validated colour-blind-safe ordering (OKLab ΔE ≥ 8 between adjacent slots under
deuteranopia/protanopia/tritanopia simulation, on a white surface).  Slots are
assigned in a fixed order and never cycled past the eighth; magnitude uses a
single-hue ramp and never a rainbow.

*Figures are drawn at their final printed size.*  ``WIDE`` and ``COLUMN`` are
the real text-block widths of the target venue — ACM Transactions on Computing
for Healthcare, i.e. ``acmart`` in its single-column ``acmsmall`` format — so a
figure included at ``\\includegraphics[width=\\linewidth]`` is not rescaled and
its labels come out at the same optical size as the body text.  This is why the
base font here is 9 pt, matching ``acmsmall`` body type, rather than the
oversized fonts a to-be-shrunk figure needs; scripts that still rely on
down-scaling can ask for ``use_paper_style(context="scaled")``.

Authoring at the true width also changes *layout* decisions, not just numbers:
5.5 inches does not hold three panels of six labelled categories side by side,
so such figures stack instead — see ``fig_gold_regime`` in the retrieval set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

#: Categorical hues, in the order they must be assigned.  Colour follows the
#: entity, never its rank: a series keeps its slot when other series are
#: filtered out.  Past eight categories, fold the tail into "Other" or facet —
#: a ninth generated hue is not separable under colour-vision deficiency.
SERIES_ORDER = ("blue", "orange", "aqua", "yellow",
                "magenta", "green", "violet", "red")

SERIES = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}

#: Chart chrome.  Grid and axis sit one shade off the surface so the data is
#: the only thing with contrast.
INK = {
    "primary": "#0b0b0b",    # titles, direct labels
    "secondary": "#52514e",  # axis labels
    "muted": "#898781",      # tick labels, annotations
    "grid": "#e1e0d9",       # hairline gridlines
    "axis": "#c3c2b7",       # baseline / spines
    "surface": "#ffffff",    # paper white
}

#: Single-hue ramp for magnitude (heatmaps, colour bars).  Light → dark, never
#: a rainbow: a multi-hue ramp implies category boundaries that are not there.
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("paper_sequential", _BLUE_RAMP)

#: Diverging ramp for signed quantities (differences, effects).  Two hues that
#: read as opposite around a neutral grey midpoint that reads as "nothing".
DIVERGING = LinearSegmentedColormap.from_list(
    "paper_diverging",
    ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8f2020"],
)

#: Marker and dash sequences, for encoding a *second* factor without spending
#: another hue.  Redundant shape/dash also keeps series apart in grayscale.
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
DASHES = ((), (4, 1.6), (1.2, 1.2), (5, 1.5, 1.2, 1.5),
          (3, 1, 1, 1, 1, 1), (7, 2), (2, 1), (1, 2))


def series(*names: str) -> list[str]:
    """Hex colours for the named slots, e.g. ``series("blue", "orange")``."""
    return [SERIES[n] for n in names]


def cycle(n: int) -> list[str]:
    """The first ``n`` categorical slots, in fixed order."""
    if n > len(SERIES_ORDER):
        raise ValueError(
            f"{n} categorical colours requested but only {len(SERIES_ORDER)} "
            "are separable under colour-vision deficiency; facet the chart or "
            "fold the tail into an 'Other' category instead."
        )
    return [SERIES[k] for k in SERIES_ORDER[:n]]


def dash(i: int) -> tuple:
    """Dash pattern for slot ``i`` — ``()`` is solid."""
    return DASHES[i % len(DASHES)]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# Target venue: ACM Transactions on Computing for Healthcare, i.e. `acmart` in
# its `acmsmall` format — a SINGLE-column layout on a 6.75x10in trim, with a
# 5.478in text block and 9pt body type.  A figure authored wider than the text
# block is scaled down by \includegraphics, shrinking its labels below the body
# text; authoring at the real width is what keeps the two matched.
WIDE = 5.478    # inches — full text width (acmsmall \textwidth)
COLUMN = 2.65   # inches — a half-width figure, two side by side
HALF = 1.75     # inches — a small inset

_GOLDEN = 0.618


def figsize(width: float = COLUMN, ratio: float = _GOLDEN) -> tuple[float, float]:
    """A figure size in inches: ``width`` wide, ``width * ratio`` tall."""
    return (width, width * ratio)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_CONTEXTS = {
    # figure drawn at its final printed size — labels land at body-text size
    "paper": dict(base=9.0, scale=1.0, line=1.4, marker=4.0),
    # figure that will still be shrunk by the LaTeX include — oversized text
    "scaled": dict(base=16.0, scale=1.0, line=1.8, marker=5.5),
    # slides / posters
    "talk": dict(base=13.0, scale=1.0, line=2.0, marker=6.0),
}


def use_paper_style(context: str = "paper") -> None:
    """Install the shared style globally.  Call once, before drawing."""
    try:
        cfg = _CONTEXTS[context]
    except KeyError:
        raise ValueError(
            f"unknown context {context!r}; expected one of {sorted(_CONTEXTS)}"
        ) from None

    base = cfg["base"]

    mpl.rcParams.update({
        # -- type ------------------------------------------------------------
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial",
                            "Nimbus Sans", "sans-serif"],
        "font.size": base,
        "axes.titlesize": base * 1.0,
        "axes.labelsize": base * 1.0,
        "xtick.labelsize": base * 0.92,
        "ytick.labelsize": base * 0.92,
        "legend.fontsize": base * 0.92,
        "figure.titlesize": base * 1.1,
        # keep maths in the same family as the labels around it
        "mathtext.fontset": "dejavusans",

        # -- colour ----------------------------------------------------------
        "axes.prop_cycle": mpl.cycler(color=cycle(len(SERIES_ORDER))),
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "text.color": INK["primary"],
        "axes.labelcolor": INK["secondary"],
        "axes.titlecolor": INK["primary"],
        "xtick.color": INK["axis"],
        "ytick.color": INK["axis"],
        "xtick.labelcolor": INK["muted"],
        "ytick.labelcolor": INK["muted"],
        "image.cmap": "paper_sequential",

        # -- chrome ----------------------------------------------------------
        # top/right spines carry no information; the remaining two are hairlines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.7,
        "axes.axisbelow": True,
        # solid hairline grid: a dashed grid reads as a threshold, not a grid
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": INK["grid"],
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        # -- marks -----------------------------------------------------------
        "lines.linewidth": cfg["line"],
        "lines.markersize": cfg["marker"],
        "lines.markeredgewidth": 0.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.0,
        "boxplot.flierprops.markersize": cfg["marker"] * 0.7,
        "scatter.edgecolors": "none",
        "hatch.linewidth": 0.7,

        # -- legend ----------------------------------------------------------
        "legend.frameon": False,
        "legend.handlelength": 1.7,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.3,

        # -- output ----------------------------------------------------------
        "figure.figsize": figsize(),
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # type 42 = embedded TrueType: required by most camera-ready checkers,
        # and it keeps the text selectable/searchable in the PDF
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 6,
    })

    # register the ramps under stable names so `cmap="paper_sequential"` works
    for cmap in (SEQUENTIAL, DIVERGING):
        try:
            mpl.colormaps.register(cmap)
        except ValueError:
            pass  # already registered by an earlier call

    # seaborn keeps its own default palette ("deep") and ignores the matplotlib
    # colour cycle, so a figure drawn with sns.boxplot would otherwise land in
    # different colours than one drawn with ax.plot
    try:
        import seaborn as sns
    except ImportError:
        pass
    else:
        sns.set_palette(cycle(len(SERIES_ORDER)))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def despine(ax, left: bool = False, bottom: bool = False) -> None:
    """Drop the left and/or bottom spine as well (e.g. for bar charts whose
    baseline is already implied by the bars)."""
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    if bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)


def grid_axis(ax, axis: str = "y") -> None:
    """Show the hairline grid on ``axis`` only (``"x"``, ``"y"`` or ``"both"``)."""
    ax.grid(False)
    ax.grid(True, axis=axis, color=INK["grid"], linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)


def panel_title(ax, letter: str, text: str = "", *, pad: float = 6.0,
                size: float | None = None) -> None:
    """Label one panel of a multi-panel figure as ``(a)``, optionally with a
    descriptive title beside it.

    Letter and text form a single *left-aligned* title.  Left alignment is what
    makes this safe: a centred title is centred on the plot box, so as soon as
    it is wider than the box — routine for a narrow panel — it overruns the
    letter beside it.

    A bare letter is set bold, as the eye needs the anchor; once a descriptive
    title follows it the whole line stays regular, so the weight never splits
    mid-title.  Every panel of a given figure takes the same form, so the two
    cases do not mix within one figure.
    """
    label = f"({letter})" + (f"  {text}" if text else "")
    ax.set_title(label, loc="left", pad=pad, color=INK["primary"],
                 fontweight="regular" if text else "bold",
                 fontsize=size or mpl.rcParams["axes.labelsize"])


def panel_titles(axes, labels: str = "abcdefgh", texts=None, **kwargs) -> None:
    """``panel_title`` applied across every panel of a figure."""
    texts = texts or [""] * len(list(axes))
    for ax, letter, text in zip(axes, labels, texts):
        panel_title(ax, letter, text, **kwargs)


def legend_above(ax, ncol: int = 3, **kwargs):
    """A frameless legend in one row above the axes, clear of the data."""
    opts = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol,
                frameon=False, borderaxespad=0.0, handletextpad=0.5)
    opts.update(kwargs)
    return ax.legend(**opts)


def value_labels(ax, bars, fmt: str = "{:.2f}", pad: float = 2.0,
                 color: str | None = None) -> None:
    """Direct-label a bar container.  Labels wear text ink, never the series
    colour — the bar beside them already carries the identity."""
    ax.bar_label(bars, fmt=fmt, padding=pad,
                 color=color or INK["secondary"],
                 fontsize=mpl.rcParams["xtick.labelsize"] * 0.95)


def save(fig, directory: Path | str, name: str, *, tight: bool = True,
         png: bool = False) -> Path:
    """Write ``fig`` to ``directory/name`` and close it.

    ``tight=False`` skips the tight bounding box, which crops mplot3d axis
    labels.  ``png=True`` additionally writes a raster copy for slide decks.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, bbox_inches="tight" if tight else None)
    if png:
        fig.savefig(path.with_suffix(".png"),
                    bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {path.parent.name}/{path.name}")
    return path
