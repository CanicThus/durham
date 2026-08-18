from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple


SNAP_HOME = "https://snap.stanford.edu/data"
DEFAULT_DATA_DIR = Path("data") / "snap"
DEFAULT_RESULT_PATH = Path("result") / "coloring_benchmark_results.json"
DEFAULT_PLOT_DIR = Path("result") / "plots"
LOCAL_TEMPLATE_GRAPH = Path("src") / "graph.txt"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    file_name: str
    description: str
    source_url: str
    local_path: Optional[Path] = None

    @property
    def download_url(self) -> str:
        return f"{SNAP_HOME}/{self.file_name}"


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    runner: Callable[["RunContext"], Dict[str, object]]
    implemented: bool = True


@dataclass(frozen=True)
class RunContext:
    graph_path: Path
    args: argparse.Namespace


def get_dataset_registry() -> Dict[str, DatasetSpec]:
    return {
        "ca-GrQc": DatasetSpec(
            name="ca-GrQc",
            file_name="ca-GrQc.txt.gz",
            description="SNAP collaboration network: Arxiv General Relativity.",
            source_url=f"{SNAP_HOME}/ca-GrQc.html",
        ),
        "facebook_combined": DatasetSpec(
            name="facebook_combined",
            file_name="facebook_combined.txt.gz",
            description="SNAP social circles from Facebook, combined ego-nets.",
            source_url=f"{SNAP_HOME}/ego-Facebook.html",
        ),
        "as20000102": DatasetSpec(
            name="as20000102",
            file_name="as20000102.txt.gz",
            description="SNAP autonomous systems graph from January 02 2000.",
            source_url=f"{SNAP_HOME}/as-733.html",
        ),
        "local-template": DatasetSpec(
            name="local-template",
            file_name=LOCAL_TEMPLATE_GRAPH.name,
            description="Local small graph for smoke tests.",
            source_url=str(LOCAL_TEMPLATE_GRAPH),
            local_path=LOCAL_TEMPLATE_GRAPH,
        ),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def quiet_call(func: Callable[[], object], verbose: bool) -> object:
    if verbose:
        return func()
    with contextlib.redirect_stdout(io.StringIO()):
        return func()


def download_file(url: str, target_path: Path, force: bool = False) -> None:
    if target_path.exists() and not force:
        return

    ensure_dir(target_path.parent)
    request = urllib.request.Request(url, headers={"User-Agent": "durhamProject/graph-coloring-test"})

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with target_path.open("wb") as output:
                shutil.copyfileobj(response, output)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Failed to download {url}. Download it manually to {target_path} "
            "or run with --datasets local-template for a local smoke test."
        ) from exc


def open_text_auto(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_edges(path: Path) -> Iterable[Tuple[int, int]]:
    with open_text_auto(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue

            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue

            if u != v:
                yield u, v


def select_nodes(edge_path: Path, max_nodes: Optional[int]) -> Optional[Set[int]]:
    if max_nodes is None or max_nodes <= 0:
        return None

    selected: Set[int] = set()
    for u, v in iter_edges(edge_path):
        if len(selected) < max_nodes:
            selected.add(u)
        if len(selected) < max_nodes:
            selected.add(v)
        if len(selected) >= max_nodes:
            break

    return selected


def write_clean_undirected_graph(
    edge_path: Path,
    output_path: Path,
    max_nodes: Optional[int],
) -> Dict[str, int]:
    selected_nodes = select_nodes(edge_path, max_nodes)
    edges: Set[Tuple[int, int]] = set()
    nodes: Set[int] = set()

    for u, v in iter_edges(edge_path):
        if selected_nodes is not None and (u not in selected_nodes or v not in selected_nodes):
            continue

        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
        nodes.add(a)
        nodes.add(b)

    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for u, v in sorted(edges):
            handle.write(f"{u} {v}\n")

    return {"nodes": len(nodes), "edges": len(edges)}


def prepare_dataset(
    spec: DatasetSpec,
    data_dir: Path,
    max_nodes: Optional[int],
    force_download: bool = False,
    force_rebuild: bool = False,
) -> Tuple[Path, Dict[str, int]]:
    ensure_dir(data_dir)

    if spec.local_path is not None:
        raw_path = spec.local_path
    else:
        raw_path = data_dir / "raw" / spec.file_name
        download_file(spec.download_url, raw_path, force=force_download)

    cache_tag = "full" if max_nodes is None or max_nodes <= 0 else f"n{max_nodes}"
    cleaned_name = f"{spec.name}.{cache_tag}.undirected.txt"
    cleaned_path = data_dir / "cleaned" / cleaned_name

    if force_rebuild or not cleaned_path.exists():
        stats = write_clean_undirected_graph(raw_path, cleaned_path, max_nodes)
    else:
        stats = graph_stats(cleaned_path)

    return cleaned_path, stats


def graph_stats(graph_path: Path) -> Dict[str, int]:
    nodes: Set[int] = set()
    edges: Set[Tuple[int, int]] = set()
    for u, v in iter_edges(graph_path):
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
        nodes.add(a)
        nodes.add(b)
    return {"nodes": len(nodes), "edges": len(edges)}


def count_conflicts(graph_path: Path, coloring: Dict[int, int]) -> int:
    conflicts = 0
    for u, v in iter_edges(graph_path):
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return conflicts


def normalize_coloring(coloring: Dict[int, int]) -> Dict[int, int]:
    return {int(node): int(color) for node, color in sorted(coloring.items())}


def run_coloring_game(ctx: RunContext) -> Dict[str, object]:
    from coloringGame import coloring_game

    def execute() -> object:
        agent = coloring_game(random_seed=ctx.args.random_seed)
        agent.load_graph(str(ctx.graph_path))
        agent.move_to_nash_equilibrium()
        return agent

    agent = quiet_call(execute, ctx.args.verbose)
    coloring = normalize_coloring(agent.node_color)
    conflicts = count_conflicts(ctx.graph_path, coloring)
    return {
        "status": "ok",
        "color_count": len(set(coloring.values())),
        "valid_coloring": conflicts == 0,
        "conflicting_edges": conflicts,
        "coloring": coloring,
    }


def run_ant_ref(ctx: RunContext) -> Dict[str, object]:
    from ant_ref import Ant_Ref

    def execute() -> Tuple[Dict[int, int], int]:
        agent = Ant_Ref(
            ants_per_iteration=ctx.args.ants_per_iteration,
            max_iterations=ctx.args.ant_iterations,
            random_seed=ctx.args.random_seed,
        )
        agent.load_graph(str(ctx.graph_path))
        return agent.solve(verbose=ctx.args.verbose)

    coloring, color_count = quiet_call(execute, ctx.args.verbose)
    coloring = normalize_coloring(coloring)
    conflicts = count_conflicts(ctx.graph_path, coloring)
    return {
        "status": "ok",
        "color_count": int(color_count),
        "valid_coloring": conflicts == 0,
        "conflicting_edges": conflicts,
        "coloring": coloring,
    }


def run_tabucol(ctx: RunContext) -> Dict[str, object]:
    from tabucal import TabuCol

    def execute() -> Tuple[Dict[int, int], int, Dict[str, int]]:
        max_colors = ctx.args.max_colors

        agent = TabuCol(
            random_seed=ctx.args.random_seed,
            max_iterations=ctx.args.tabu_iterations,
            max_restarts=ctx.args.tabu_restarts,
        )
        agent.load_graph(str(ctx.graph_path))
        coloring, color_count = agent.solve(
            max_colors=max_colors,
            min_colors=ctx.args.min_colors,
            max_iterations=ctx.args.tabu_iterations,
            max_restarts=ctx.args.tabu_restarts,
            verbose=ctx.args.verbose,
        )
        return coloring, color_count, dict(agent.last_search_stats)

    coloring, color_count, search_stats = quiet_call(execute, ctx.args.verbose)
    coloring = normalize_coloring(coloring)
    conflicts = count_conflicts(ctx.graph_path, coloring)
    return {
        "status": "ok",
        "color_count": int(color_count),
        "valid_coloring": conflicts == 0,
        "conflicting_edges": conflicts,
        "coloring": coloring,
        "search_stats": search_stats,
    }


def run_dsatur(ctx: RunContext) -> Dict[str, object]:
    from dsatur import DSATUR

    def execute() -> Tuple[Dict[int, int], int]:
        agent = DSATUR()
        agent.load_graph(str(ctx.graph_path))
        return agent.solve(verbose=ctx.args.verbose)

    coloring, color_count = quiet_call(execute, ctx.args.verbose)
    coloring = normalize_coloring(coloring)
    conflicts = count_conflicts(ctx.graph_path, coloring)
    return {
        "status": "ok",
        "color_count": int(color_count),
        "valid_coloring": conflicts == 0,
        "conflicting_edges": conflicts,
        "coloring": coloring,
    }


def get_algorithm_registry() -> Dict[str, AlgorithmSpec]:
    return {
        "coloring_game": AlgorithmSpec("coloring_game", run_coloring_game),
        "ant_ref": AlgorithmSpec("ant_ref", run_ant_ref),
        "tabucol": AlgorithmSpec("tabucol", run_tabucol),
        "dsatur": AlgorithmSpec("dsatur", run_dsatur),
    }


def run_algorithm(algorithm: AlgorithmSpec, ctx: RunContext) -> Dict[str, object]:
    start = time.perf_counter()

    try:
        result = algorithm.runner(ctx)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "algorithm": algorithm.name,
            "status": "error",
            "execution_time_sec": round(elapsed, 6),
            "color_count": None,
            "valid_coloring": None,
            "conflicting_edges": None,
            "coloring": {},
            "error": f"{type(exc).__name__}: {exc}",
        }

    elapsed = time.perf_counter() - start
    return {
        "algorithm": algorithm.name,
        "execution_time_sec": round(elapsed, 6),
        **result,
    }


def print_algorithm_result(result: Dict[str, object], args: argparse.Namespace) -> None:
    print(
        f"  {result['algorithm']}: "
        f"status={result['status']}, "
        f"time={result['execution_time_sec']}s, "
        f"colors={result['color_count']}, "
        f"valid={result['valid_coloring']}, "
        f"conflicts={result['conflicting_edges']}"
    )

    coloring = result.get("coloring") or {}
    if args.print_coloring:
        if len(coloring) <= args.coloring_print_limit:
            print(f"    coloring={coloring}")
        else:
            print(
                f"    coloring has {len(coloring)} vertices; "
                "full mapping is saved in the JSON result file."
            )

    if result.get("message"):
        print(f"    message={result['message']}")
    if result.get("error"):
        print(f"    error={result['error']}")


def safe_file_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def plot_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    value_format: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)

    fig_width = max(7.0, 1.4 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    bars = ax.bar(labels, values, color="#2f6f8f")

    ax.set_title(title)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_benchmark_images(report: Dict[str, object], plot_dir: Path) -> List[Path]:
    generated: List[Path] = []

    for dataset_report in report["datasets"]:
        dataset_name = str(dataset_report["dataset"])
        dataset_stem = safe_file_stem(dataset_name)
        results = dataset_report["results"]

        time_items = [
            (str(result["algorithm"]), float(result["execution_time_sec"]))
            for result in results
            if isinstance(result.get("execution_time_sec"), (int, float))
        ]
        if time_items:
            labels, values = zip(*time_items)
            output_path = plot_dir / f"{dataset_stem}_algorithm_time.png"
            plot_bar_chart(
                labels=list(labels),
                values=list(values),
                title=f"{dataset_name} Algorithm Execution Time",
                ylabel="Time (seconds)",
                output_path=output_path,
                value_format="{:.4f}",
            )
            generated.append(output_path)

        color_items = [
            (str(result["algorithm"]), float(result["color_count"]))
            for result in results
            if isinstance(result.get("color_count"), (int, float))
        ]
        if color_items:
            labels, values = zip(*color_items)
            output_path = plot_dir / f"{dataset_stem}_algorithm_color_count.png"
            plot_bar_chart(
                labels=list(labels),
                values=list(values),
                title=f"{dataset_name} Final Color Count",
                ylabel="Color count",
                output_path=output_path,
                value_format="{:.0f}",
            )
            generated.append(output_path)

    return generated


def parse_args() -> argparse.Namespace:
    dataset_registry = get_dataset_registry()
    algorithm_registry = get_algorithm_registry()

    parser = argparse.ArgumentParser(
        description="Benchmark graph coloring algorithms on Stanford SNAP datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ca-GrQc", "facebook_combined", "as20000102"],
        choices=sorted(dataset_registry),
        help="Dataset names to run.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(algorithm_registry),
        choices=sorted(algorithm_registry),
        help="Algorithms to run. Empty algorithm files are reserved as not_implemented slots.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1000,
        help="Use an induced subgraph from the first N SNAP nodes. Use 0 for full graph.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ant-iterations", type=int, default=20)
    parser.add_argument("--ants-per-iteration", type=int, default=10)
    parser.add_argument("--tabu-iterations", type=int, default=2000)
    parser.add_argument("--tabu-restarts", type=int, default=5)
    parser.add_argument("--max-colors", type=int, default=None)
    parser.add_argument("--min-colors", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-coloring", action="store_true")
    parser.add_argument("--coloring-print-limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_registry = get_dataset_registry()
    algorithm_registry = get_algorithm_registry()
    max_nodes = None if args.max_nodes <= 0 else args.max_nodes

    ensure_dir(args.result_path.parent)

    report: Dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": "Stanford Large Network Dataset Collection",
        "data_source_url": SNAP_HOME,
        "max_nodes": max_nodes,
        "datasets": [],
    }

    for dataset_name in args.datasets:
        spec = dataset_registry[dataset_name]
        graph_path, stats = prepare_dataset(
            spec,
            data_dir=args.data_dir,
            max_nodes=max_nodes,
            force_download=args.force_download,
            force_rebuild=args.force_rebuild,
        )

        dataset_report: Dict[str, object] = {
            "dataset": spec.name,
            "description": spec.description,
            "source_url": spec.source_url,
            "graph_path": str(graph_path),
            "nodes": stats["nodes"],
            "edges": stats["edges"],
            "results": [],
        }

        print(
            f"Dataset {spec.name}: nodes={stats['nodes']}, "
            f"edges={stats['edges']}, graph={graph_path}"
        )

        ctx = RunContext(graph_path=graph_path, args=args)
        for algorithm_name in args.algorithms:
            algorithm = algorithm_registry[algorithm_name]
            result = run_algorithm(algorithm, ctx)
            dataset_report["results"].append(result)
            print_algorithm_result(result, args)

        report["datasets"].append(dataset_report)

    with args.result_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Full benchmark results saved to {args.result_path}")

    if not args.no_plots:
        generated_plots = plot_benchmark_images(report, args.plot_dir)
        for plot_path in generated_plots:
            print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
