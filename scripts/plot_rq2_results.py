#!/usr/bin/env python3
"""Refresh and serve the RQ2 interactive HTML plot."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


class PlotPageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/rq2-results-figures.html"):
            self.send_error(404)
            return
        content = (ROOT / "rq2-results-figures.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    args = parser.parse_args()

    write_html()
    with ThreadingHTTPServer((args.host, args.port), PlotPageHandler) as server:
        url_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
        print(f"Serving RQ2 results at http://{url_host}:{args.port}/rq2-results-figures.html")
        print("Press Ctrl-C to stop the server.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
