#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The perf-llm Project Authors

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_label_map(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    mapping: dict[str, str] = {}
    for item in value.split(","):
        key, sep, mapped = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise argparse.ArgumentTypeError("--labels must use the form id=label,id=label")
        mapping[key] = mapped.strip()
    return mapping


def load_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, path_str in enumerate(paths, start=1):
        path = Path(path_str)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row_with_source = dict(row)
                row_with_source["source_id"] = str(index)
                rows.append(row_with_source)
    return rows


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def line_key(row: dict[str, str]) -> tuple[str, str]:
    model = row.get("model", "")
    model_tail = model.split("/")[-1] if model else ""
    source_id = row.get("source_id", "")
    return source_id, model_tail


def format_line_label(
    source_id: str, model_tail: str, label_map: dict[str, str] | None = None
) -> str:
    source_label = source_id
    if label_map is not None:
        source_label = label_map.get(source_id, "")
    if source_label:
        return f"[{source_label}] {model_tail}"
    return model_tail


def require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for perf-llm-display. Install with: pip install '.[viz]'"
        ) from exc
    return plt


def create_figure(plt: Any) -> tuple[Any, Any]:
    return plt.subplots(figsize=(10.24, 7.68), dpi=100)


def shift_x_values(x_values: list[int], series_index: int, series_count: int) -> list[float]:
    if series_count <= 1:
        return [float(x) for x in x_values]
    span = 0.24
    step = span / (series_count - 1)
    offset = -span / 2 + series_index * step
    return [x + offset for x in x_values]


def configure_concurrency_axis(ax: Any, values: set[int], *, alpha: float = 1.0) -> None:
    ax.set_xticks(sorted(values))
    ax.grid(True, axis="y", alpha=alpha)
    ax.grid(False, axis="x")


def plot_ttft(args: argparse.Namespace) -> int:
    rows = load_rows(args.csv)
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    label_map = parse_label_map(args.labels)

    for row in rows:
        concurrency = parse_int(row.get("concurrency"))
        ttft = parse_float(row.get("first_token_latency_mean_s"))
        if concurrency is None or ttft is None:
            continue
        grouped[line_key(row)].append((concurrency, ttft))

    if not grouped:
        raise SystemExit("No ttft data found in CSV")

    plt = require_matplotlib()
    fig, ax = create_figure(plt)
    sorted_groups = sorted(grouped.items())
    concurrency_values: set[int] = set()
    for series_index, ((source_id, model_tail), points) in enumerate(sorted_groups):
        points.sort(key=lambda item: item[0])
        x = [item[0] for item in points]
        concurrency_values.update(x)
        x_shifted = shift_x_values(x, series_index, len(sorted_groups))
        y = [item[1] for item in points]
        ax.plot(
            x_shifted,
            y,
            marker="o",
            label=format_line_label(source_id, model_tail, label_map),
        )

    ax.set_title("Time to first token vs concurrency\n(lower is better)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TTFT mean (s)")
    configure_concurrency_axis(ax, concurrency_values)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output)
    else:
        plt.show()
    return 0


def plot_tps(args: argparse.Namespace) -> int:
    rows = load_rows(args.csv)
    grouped_mean: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    grouped_wall: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    label_map = parse_label_map(args.labels)

    for row in rows:
        concurrency = parse_int(row.get("concurrency"))
        tps_mean = parse_float(row.get("throughput_mean_tokens_s"))
        tps_wall = parse_float(row.get("throughput_wall_tokens_s"))
        if concurrency is None:
            continue
        key = line_key(row)
        if tps_mean is not None:
            grouped_mean[key].append((concurrency, tps_mean))
        if tps_wall is not None:
            grouped_wall[key].append((concurrency, tps_wall))

    if not grouped_mean and not grouped_wall:
        raise SystemExit("No throughput data found in CSV")

    plt = require_matplotlib()
    fig, ax = create_figure(plt)

    series: list[tuple[str, tuple[str, str], list[tuple[int, float]]]] = []
    concurrency_values: set[int] = set()
    for key, points in sorted(grouped_mean.items()):
        series.append(("tps_mean", key, points))
    for key, points in sorted(grouped_wall.items()):
        series.append(("tps_wall", key, points))

    for series_index, (metric, (source_id, model_tail), points) in enumerate(series):
        points.sort(key=lambda item: item[0])
        x = [item[0] for item in points]
        concurrency_values.update(x)
        x_shifted = shift_x_values(x, series_index, len(series))
        y = [item[1] for item in points]
        label = format_line_label(source_id, model_tail, label_map)
        if metric == "tps_mean":
            ax.plot(x_shifted, y, marker="o", label=f"{label} tps_mean")
        else:
            ax.plot(x_shifted, y, marker="x", linestyle="--", label=f"{label} tps_wall")

    ax.set_title("Throughput vs concurrency\n(higher is better)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Tokens / second")
    configure_concurrency_axis(ax, concurrency_values)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output)
    else:
        plt.show()
    return 0


def plot_lat(args: argparse.Namespace) -> int:
    rows = load_rows(args.csv)
    label_map = parse_label_map(args.labels)
    normalize = args.normalize

    # Group data by file id/model, then aggregate all metrics for each concurrency
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        concurrency = parse_int(row.get("concurrency"))
        if concurrency is None:
            continue
        grouped[line_key(row)].append(
            {
                "concurrency": concurrency,
                "min": parse_float(row.get("latency_min_s")),
                "mean": parse_float(row.get("latency_mean_s")),
                "p50": parse_float(row.get("latency_p50_s")),
                "max": parse_float(row.get("latency_max_s")),
            }
        )

    if not grouped:
        raise SystemExit("No latency data found in CSV")

    plt = require_matplotlib()
    fig, ax = create_figure(plt)
    sorted_groups = sorted(grouped.items())
    concurrency_values: set[int] = set()

    for series_index, ((source_id, model_tail), data_points) in enumerate(sorted_groups):
        # Aggregate by concurrency value
        agg_by_conc: dict[int, dict[str, float | None]] = {}
        for p in data_points:
            conc_val = int(p["concurrency"])
            if conc_val not in agg_by_conc:
                agg_by_conc[conc_val] = {
                    "min": p["min"],
                    "mean": p["mean"],
                    "p50": p["p50"],
                    "max": p["max"],
                }
            else:
                # Keep first non-None value for each metric type per concurrency
                entry = agg_by_conc[conc_val]
                if p["min"] is not None and entry["min"] is None:
                    entry["min"] = p["min"]
                if p["mean"] is not None and entry["mean"] is None:
                    entry["mean"] = p["mean"]
                if p["p50"] is not None and entry["p50"] is None:
                    entry["p50"] = p["p50"]
                if p["max"] is not None and entry["max"] is None:
                    entry["max"] = p["max"]

        if not agg_by_conc:
            continue

        baseline_entry = agg_by_conc.get(1)
        baseline_mean = None if baseline_entry is None else baseline_entry["mean"]
        if normalize and (baseline_mean is None or baseline_mean == 0):
            continue
        normalized_baseline_mean = 1.0 if baseline_mean is None else float(baseline_mean)

        # Extract sorted values
        concurrencies = sorted(agg_by_conc.keys())
        x: list[int] = []
        y_mean: list[float] = []
        y_error_lo: list[float] = []  # lower error (mean - min)
        y_error_hi: list[float] = []  # upper error (max - mean)
        y_p50: list[float] = []

        scale: float = 1.0 if not normalize else normalized_baseline_mean

        for conc in concurrencies:
            entry = agg_by_conc[conc]
            mean_val = entry["mean"]

            if mean_val is None:
                continue

            x.append(conc)
            y_mean.append(mean_val / scale)

            # Calculate distances from mean for error bars (positive values)
            min_val = entry["min"]
            max_val = entry["max"]

            if min_val is not None:
                y_error_lo.append(abs(mean_val - min_val) / scale)  # lower direction
            else:
                y_error_lo.append(0.0)

            if max_val is not None:
                y_error_hi.append(abs(max_val - mean_val) / scale)  # upper direction
            else:
                y_error_hi.append(0.0)

            p50_val = entry["p50"]
            if p50_val is not None:
                y_p50.append(p50_val / scale)

        if not x or not y_mean:
            continue

        concurrency_values.update(x)
        x_shifted = shift_x_values(x, series_index, len(sorted_groups))

        # Plot mean as the main line with circle markers (solid)
        label = format_line_label(source_id, model_tail, label_map)
        ax.plot(x_shifted, y_mean, marker="o", linestyle="-", label=label)

        # Get color of the line just plotted
        line_color = fig.gca().lines[-1].get_color()

        # Plot error bars for min/max range (transparent fill)
        ax.errorbar(
            x_shifted,
            y_mean,
            yerr=[y_error_lo, y_error_hi],
            fmt="none",
            ecolor=line_color,
            capsize=4,
            alpha=0.5,
        )

        # Plot median (p50) as X markers on top of the line (no legend entry)
        if y_p50 and len(y_p50) == len(x):
            ax.plot(
                x_shifted,
                y_p50,
                marker="x",
                linestyle="",
                color=line_color,
                label="_nolegend_",
                alpha=0.7,
                zorder=3,
            )

    title = "Latency vs concurrency"
    if normalize:
        title += " normalized to concurrency 1"
    ax.set_title(f"{title}\n(lower is better)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Latency / latency@1" if normalize else "Latency (s)")
    configure_concurrency_axis(ax, concurrency_values, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output)
    else:
        plt.show()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display figures from perf-llm CSV outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ttft_parser = subparsers.add_parser(
        "ttft",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Draw time-to-first-token as a function of concurrency",
    )
    ttft_parser.add_argument(
        "csv",
        nargs="+",
        help="One or more input CSV files generated by perf-llm",
    )
    ttft_parser.add_argument("--output", help="Write figure to this file instead of showing it")
    ttft_parser.add_argument(
        "--labels",
        help="Map input file ids to legend labels, e.g. 1=run-a,2=run-b",
    )

    tps_parser = subparsers.add_parser(
        "tps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Draw throughput as a function of concurrency",
    )
    tps_parser.add_argument(
        "csv",
        nargs="+",
        help="One or more input CSV files generated by perf-llm",
    )
    tps_parser.add_argument("--output", help="Write figure to this file instead of showing it")
    tps_parser.add_argument(
        "--labels",
        help="Map input file ids to legend labels, e.g. 1=run-a,2=run-b",
    )

    lat_parser = subparsers.add_parser(
        "lat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Draw latency percentiles as error bars and mean as points",
    )
    lat_parser.add_argument(
        "csv",
        nargs="+",
        help="One or more input CSV files generated by perf-llm",
    )
    lat_parser.add_argument("--output", help="Write figure to this file instead of showing it")
    lat_parser.add_argument(
        "--labels",
        help="Map input file ids to legend labels, e.g. 1=run-a,2=run-b",
    )
    lat_parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize latency values to the concurrency=1 mean for each series",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "ttft":
        return plot_ttft(args)
    if args.command == "tps":
        return plot_tps(args)
    if args.command == "lat":
        return plot_lat(args)
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
