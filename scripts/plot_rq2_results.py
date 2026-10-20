#!/usr/bin/env python3
"""Generate RQ2 HTML and SVG figures from saved result files."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "dvae": "#3367d6",
    "varlen_dvae": "#0f9d58",
    "rkmeans": "#f4b400",
    "rqvae": "#9c27b0",
    "varlen_rqvae": "#e91e63",
}
LABELS = {
    "dvae": "dVAE",
    "varlen_dvae": "Varlen dVAE",
    "rkmeans": "RKMeans",
    "rqvae": "RQ-VAE",
    "varlen_rqvae": "Varlen RQ-VAE",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_chart(title, values, width=620, height=330):
    left, top, right, bottom = 70, 48, 24, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(values.values()) or 1.0
    maximum *= 1.12
    parts = [f'<g font-family="Arial, sans-serif">', f'<text x="{width/2}" y="25" text-anchor="middle" font-size="18" font-weight="bold">{title}</text>']
    for i in range(6):
        y = top + plot_h - plot_h * i / 5
        val = maximum * i / 5
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d9dfe7"/>')
        parts.append(f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" font-size="11">{val:.3f}</text>')
    n = len(values)
    slot = plot_w / n
    bar_w = min(105, slot * 0.62)
    for i, (method, value) in enumerate(values.items()):
        x = left + slot * i + (slot - bar_w) / 2
        h = plot_h * value / maximum
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{COLORS[method]}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-size="11">{value:.4f}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="12">{LABELS[method]}</text>')
    parts.append(f'<line x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}" stroke="#555"/>')
    parts.append("</g>")
    return "".join(parts)


def line_chart(title, series, width=520, height=380, y_label="Metric value"):
    left, top, right, bottom = 62, 80, 24, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = sorted({x for points in series.values() for x, _ in points})
    if not xs:
        return f'<g font-family="Arial, sans-serif"><text x="{width/2}" y="25" text-anchor="middle" font-size="18" font-weight="bold">{title}</text><text x="{width/2}" y="{height/2}" text-anchor="middle" font-size="13">No per-length dVAE seqrec.json results available</text></g>'
    maximum = max((y for points in series.values() for _, y in points), default=1.0) or 1.0
    maximum *= 1.12
    parts = [
        '<g font-family="Arial, sans-serif">',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-size="18" font-weight="bold">{title}</text>',
    ]
    for i in range(6):
        y = top + plot_h - plot_h * i / 5
        val = maximum * i / 5
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d9dfe7"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{val:.3f}</text>')
    x_min, x_max = min(xs), max(xs)
    x_pos = lambda x: left + (plot_w / 2 if x_min == x_max else plot_w * (x - x_min) / (x_max - x_min))
    y_pos = lambda y: top + plot_h - plot_h * y / maximum
    last_tick_x = float("-inf")
    for x in xs:
        px = x_pos(x)
        if px - last_tick_x < 30:
            continue
        last_tick_x = px
        label = str(int(x)) if float(x).is_integer() else f"{x:.2f}"
        parts.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{top+plot_h}" y2="{top+plot_h+4}" stroke="#555"/>')
        parts.append(f'<text x="{px:.1f}" y="{height-bottom+20}" text-anchor="middle" font-size="11">{label}</text>')
    colors = {
        "dVAE": "#3367d6",
        "RKMeans": "#f4b400",
        "RQ-VAE": "#9c27b0",
        "Varlen dVAE": "#0f9d58",
        "Varlen RQ-VAE": "#e91e63",
    }
    for name, points in series.items():
        tokenizer = name.split("@", 1)[0]
        if tokenizer.startswith("Varlen dVAE"):
            tokenizer = "Varlen dVAE"
        elif tokenizer.startswith("Varlen RQ-VAE"):
            tokenizer = "Varlen RQ-VAE"
        maxlen_match = re.search(r"maxlen (\d+)", name)
        varlen_maxlen = int(maxlen_match.group(1)) if maxlen_match else None
        color = colors[tokenizer]
        coords = " ".join(f'{x_pos(x):.1f},{y_pos(y):.1f}' for x, y in sorted(points))
        if not tokenizer.startswith("Varlen") and len(points) > 1:
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, value in sorted(points):
            px, py = x_pos(x), y_pos(value)
            if tokenizer.startswith("Varlen") and varlen_maxlen == 5:
                marker = f'<path d="M {px:.1f} {py-5:.1f} L {px+5:.1f} {py:.1f} L {px:.1f} {py+5:.1f} L {px-5:.1f} {py:.1f} Z" fill="{color}"/>'
            elif tokenizer.startswith("Varlen") and varlen_maxlen == 9:
                marker = f'<path d="M {px:.1f} {py-5:.1f} L {px+5:.1f} {py+4:.1f} L {px-5:.1f} {py+4:.1f} Z" fill="{color}"/>'
            else:
                marker = f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>'
            parts.append(f'<g>{marker}<title>{name}, mean length {x}: {value:.5f}</title></g>')
    parts.extend([
        f'<line x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}" stroke="#555"/>',
        f'<text x="{left+plot_w/2}" y="{height-12}" text-anchor="middle" font-size="12">Semantic ID length</text>',
        f'<text transform="translate(16,{top+plot_h/2}) rotate(-90)" text-anchor="middle" font-size="12">{y_label}</text>',
    ])
    legend_x = left + 12
    for i, tokenizer in enumerate(("dVAE", "RKMeans", "RQ-VAE")):
        x = legend_x + i * 125
        color = colors[tokenizer]
        parts.append(f'<line x1="{x}" x2="{x+20}" y1="42" y2="42" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{x+26}" y="46" font-size="11">{tokenizer}</text>')
    for i, name in enumerate(n for n in series if n.startswith("Varlen")):
        x = legend_x + i * 215
        varlen_tokenizer = "Varlen RQ-VAE" if name.startswith("Varlen RQ-VAE") else "Varlen dVAE"
        color = colors[varlen_tokenizer]
        maxlen_match = re.search(r"maxlen (\d+)", name)
        varlen_maxlen = int(maxlen_match.group(1)) if maxlen_match else None
        if varlen_maxlen == 5:
            marker = f'<path d="M {x+4} 57 L {x+9} 62 L {x+4} 67 L {x-1} 62 Z" fill="{color}"/>'
        elif varlen_maxlen == 9:
            marker = f'<path d="M {x+4} 57 L {x+9} 66 L {x-1} 66 Z" fill="{color}"/>'
        else:
            marker = f'<circle cx="{x+4}" cy="62" r="4" fill="{color}"/>'
        parts.append(marker)
        parts.append(f'<text x="{x+14}" y="66" font-size="11">{name.split("@", 1)[0]}</text>')
    parts.append('</g>')
    return "".join(parts)


TOKENIZER_DIRS = {
    "dvae": "dvae",
    "rkmeans": "rkmeans",
    "rqvae": "rqvae",
    "varlen_dvae": "dvae_varlen_3",
    "varlen_rqvae": "rqvae_varlen_3",
}

VARLEN_LABELS = {
    "varlen_dvae": "Varlen dVAE",
    "varlen_rqvae": "Varlen RQ-VAE",
}


def tokenizer_length_series(base, tokenizer, metric, cutoff):
    metric = metric.lower()
    series = {}
    for path in sorted((base / TOKENIZER_DIRS[tokenizer]).glob("maxlen_*/seqrec.json")):
        maxlen = int(path.parent.name.removeprefix("maxlen_"))
        if tokenizer in VARLEN_LABELS:
            metrics_file = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
            length = float(metrics_file["train_holdout"]["length"]["mean_uniform"])
            name = f"{VARLEN_LABELS[tokenizer]} maxlen {maxlen}@{cutoff}"
        else:
            length = maxlen
            name = f"{tokenizer}@{cutoff}"
        with path.open(encoding="utf-8") as f:
            metrics = json.load(f)["metrics"]
        series.setdefault(name, []).append((length, float(metrics[f"{metric}@{cutoff}"])))
    return series


def combined_length_series(base, metric, cutoff):
    display = {"dvae": "dVAE", "rkmeans": "RKMeans", "rqvae": "RQ-VAE"}
    combined = {}
    for tokenizer in TOKENIZER_DIRS:
        for name, points in tokenizer_length_series(base, tokenizer, metric, cutoff).items():
            if tokenizer in display:
                name = name.replace(tokenizer, display[tokenizer], 1)
            combined[name] = points
    return combined


def load_plot_data(dataset):
    """Load plotted values from per-length tokenizer seqrec files."""
    base = ROOT / "results" / "RQ2" / dataset
    data = {}
    for tokenizer, directory in TOKENIZER_DIRS.items():
        runs = sorted((base / directory).glob("maxlen_*/seqrec.json"), key=lambda p: int(p.parent.name.removeprefix("maxlen_")))
        model = {"lengths": [], "maxlens": [], "length_distributions": [], **{f"{metric}{cutoff}": [] for metric in ("recall", "ndcg", "coverage") for cutoff in (10, 100)}}
        for path in runs:
            maxlen = int(path.parent.name.removeprefix("maxlen_"))
            if tokenizer in VARLEN_LABELS:
                metrics_file = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
                length = float(metrics_file["train_holdout"]["length"]["mean_uniform"])
                counts = metrics_file["train_holdout"]["popularity_length"]["pop_stats_by_length"]
                model["length_distributions"].append({
                    "maxlen": maxlen,
                    "lengths": list(range(1, maxlen + 1)),
                    "counts": [int(counts.get(str(i), {}).get("count", 0)) for i in range(1, maxlen + 1)],
                })
            else:
                length = maxlen
            metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
            model["lengths"].append(length)
            model["maxlens"].append(maxlen)
            for metric in ("recall", "ndcg", "coverage"):
                for cutoff in (10, 100):
                    model[f"{metric}{cutoff}"].append(float(metrics[f"{metric}@{cutoff}"]))
        data[tokenizer] = model
    return data


def write_html():
    """Refresh the interactive page's embedded data from result JSON files."""
    path = ROOT / "rq2-results-figures.html"
    html = path.read_text(encoding="utf-8")
    datasets = {name: load_plot_data(name) for name in ("instruments", "video_games", "yambda")}
    replacement = "/* GENERATED_DATA_START */\n      const data = " + json.dumps(datasets, indent=8) + ";\n      /* GENERATED_DATA_END */"
    pattern = r"/\* GENERATED_DATA_START \*/.*?/\* GENERATED_DATA_END \*/"
    updated, count = re.subn(pattern, lambda _: replacement, html, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Expected one generated data block in {path}")
    path.write_text(updated, encoding="utf-8")


def length_distribution_chart(base, width=900, height=360):
    runs = []
    for tokenizer in VARLEN_LABELS:
        directory = base / TOKENIZER_DIRS[tokenizer]
        paths = sorted(directory.glob("maxlen_*/seqrec.json"), key=lambda p: int(p.parent.name.removeprefix("maxlen_")))
        for path in paths:
            maxlen = int(path.parent.name.removeprefix("maxlen_"))
            metrics = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
            by_length = metrics["train_holdout"]["popularity_length"]["pop_stats_by_length"]
            counts = [int(by_length.get(str(length), {}).get("count", 0)) for length in range(1, maxlen + 1)]
            runs.append({"tokenizer": tokenizer, "maxlen": maxlen, "counts": counts, "total": sum(counts) or 1})
    if not runs:
        return f'<g font-family="Arial, sans-serif"><text x="{width/2}" y="{height/2}" text-anchor="middle" font-size="13">No variable-length distributions available</text></g>'

    left, top, right, bottom = 62, 70, 24, 52
    plot_w, plot_h = width - left - right, height - top - bottom
    max_length = max(run["maxlen"] for run in runs)
    group_width = plot_w / max_length
    bar_width = min(30, group_width * 0.7 / len(runs))
    parts = [
        '<g font-family="Arial, sans-serif">',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-weight="bold">Variable-length allocated length distribution</text>',
    ]
    for tick in range(5):
        percent = tick * 25
        y = top + plot_h - plot_h * percent / 100
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d9dfe7"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{percent}%</text>')
    for length in range(1, max_length + 1):
        center = left + group_width * (length - 0.5)
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+20}" text-anchor="middle" font-size="11">{length}</text>')
        for i, run in enumerate(runs):
            count = run["counts"][length - 1] if length <= run["maxlen"] else 0
            percent = 100 * count / run["total"]
            bar_h = plot_h * percent / 100
            x = center + (i - (len(runs) - 1) / 2) * (bar_width + 2) - bar_width / 2
            y = top + plot_h - bar_h
            color = COLORS[run["tokenizer"]]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="{color}"><title>maxlen {run["maxlen"]}, allocated length {length}: {count} items ({percent:.1f}%)</title></rect>')
    parts.extend([
        f'<text x="{left+plot_w/2}" y="{height-8}" text-anchor="middle" font-size="12">Allocated semantic ID length</text>',
        f'<text transform="translate(16,{top+plot_h/2}) rotate(-90)" text-anchor="middle" font-size="12">Train-holdout items (%)</text>',
    ])
    for i, run in enumerate(runs):
        x = left + 10 + i * 130
        color = COLORS[run["tokenizer"]]
        label = LABELS[run["tokenizer"]]
        parts.append(f'<rect x="{x}" y="43" width="12" height="12" fill="{color}"/><text x="{x+18}" y="53" font-size="11">{label} maxlen {run["maxlen"]}</text>')
    parts.append("</g>")
    return "".join(parts)


def write_figure(dataset):
    base = ROOT / "results" / "RQ2" / dataset
    sid = read_csv(base / "semantic_id_comparison.csv")
    rec = read_csv(base / "seqrec_comparison.csv")
    sid_train = {r["method"]: float(r["recon_at_last"]) for r in sid if r["split"] == "train_holdout"}
    lengths = {r["method"]: float(r["mean_length"]) for r in sid if r["split"] == "train_holdout"}
    rec1 = {r["method"]: float(r["recall@1"]) for r in rec}
    rec10 = {r["method"]: float(r["recall@10"]) for r in rec}
    rec100 = {r["method"]: float(r["recall@100"]) for r in rec}
    ndcg10 = {r["method"]: float(r["ndcg@10"]) for r in rec}
    ndcg100 = {r["method"]: float(r["ndcg@100"]) for r in rec}
    coverage10 = {r["method"]: float(r["coverage@10"]) for r in rec}
    coverage100 = {r["method"]: float(r["coverage@100"]) for r in rec}
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1940" viewBox="0 0 1600 1940">
<rect width="100%" height="100%" fill="white"/>
<text x="800" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold">RQ2 — {dataset}</text>
<g transform="translate(0,55)">{bar_chart("Semantic reconstruction", sid_train, width=390)}</g>
<g transform="translate(0,425)">{bar_chart("Mean semantic-ID length", lengths, width=390)}</g>
<g transform="translate(400,55)">{bar_chart("Recall@10", rec10, width=390)}</g>
<g transform="translate(400,425)">{bar_chart("Recall@100", rec100, width=390)}</g>
<g transform="translate(800,55)">{bar_chart("NDCG@10", ndcg10, width=390)}</g>
<g transform="translate(800,425)">{bar_chart("NDCG@100", ndcg100, width=390)}</g>
<g transform="translate(1200,55)">{bar_chart("Coverage@10", coverage10, width=390)}</g>
<g transform="translate(1200,425)">{bar_chart("Coverage@100", coverage100, width=390)}</g>
<g transform="translate(0,795)">{line_chart("Recall@10 by semantic ID length", combined_length_series(base, "recall", 10), width=520, height=380, y_label="Recall")}</g>
<g transform="translate(520,795)">{line_chart("NDCG@10 by semantic ID length", combined_length_series(base, "ndcg", 10), width=520, height=380, y_label="NDCG")}</g>
<g transform="translate(1040,795)">{line_chart("Coverage@10 by semantic ID length", combined_length_series(base, "coverage", 10), width=520, height=380, y_label="Coverage (items)")}</g>
<g transform="translate(0,1175)">{line_chart("Recall@100 by semantic ID length", combined_length_series(base, "recall", 100), width=520, height=380, y_label="Recall")}</g>
<g transform="translate(520,1175)">{line_chart("NDCG@100 by semantic ID length", combined_length_series(base, "ndcg", 100), width=520, height=380, y_label="NDCG")}</g>
<g transform="translate(1040,1175)">{line_chart("Coverage@100 by semantic ID length", combined_length_series(base, "coverage", 100), width=520, height=380, y_label="Coverage (items)")}</g>
<g transform="translate(350,1555)">{length_distribution_chart(base)}</g>
</svg>'''
    (base / "rq2_results_figure.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    write_html()
