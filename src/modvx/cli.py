"""
Command-line interface for modvx.

Provides the ``modvx`` entry-point with subcommands:

* ``modvx run``          — execute the full FSS computation pipeline.
* ``modvx extract-csv``  — extract FSS NetCDF results to per-experiment CSVs.
* ``modvx plot``         — generate FSS-vs-leadtime plots.
* ``modvx validate``     — list available domains / thresholds / windows.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .config import ModvxConfig, load_config, merge_cli_overrides
from .utils import parse_datetime


def _setup_logging(cfg: ModvxConfig) -> None:
    """Configure root logging so that logger.info messages are visible on the console.
    The ``run`` subcommand relies on TaskManager to do this; other subcommands bypass
    TaskManager entirely, so this helper fills the gap."""
    level = logging.DEBUG if cfg.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Register arguments that are common to all modvx subcommands on a given parser.
    Currently registers the ``--config`` flag for pointing to a YAML configuration file
    and ``--verbose`` for enabling detailed debug logging. Centralising these shared
    arguments avoids repetition across _build_run_parser, _build_extract_parser,
    _build_plot_parser, and _build_validate_parser.

    Parameters:
        parser (argparse.ArgumentParser): Argument parser instance for a specific subcommand.

    Returns:
        None
    """
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: use built-in defaults).",
    )
    parser.add_argument("--verbose", action="store_true", default=None)


# ======================================================================
# Subcommand: run
# ======================================================================

def _build_run_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="Execute the FSS computation pipeline.")
    _add_common_args(p)

    p.add_argument("--expname", dest="experiment_name", type=str, default=None)
    p.add_argument("--start", dest="initial_cycle_start", type=parse_datetime, default=None)
    p.add_argument("--end", dest="final_cycle_start", type=parse_datetime, default=None)
    p.add_argument("--forecast-step", dest="forecast_step_hours", type=int, default=None)
    p.add_argument("--obs-interval", dest="observation_interval_hours", type=int, default=None)
    p.add_argument("--cycle-interval", dest="cycle_interval_hours", type=int, default=None)
    p.add_argument("--forecast-length", dest="forecast_length_hours", type=int, default=None)
    p.add_argument("--target-resolution", dest="target_resolution", type=str, default=None)
    p.add_argument("--vxdomain", dest="vxdomain", type=str, default=None,
                   help="Comma-separated verification domains (e.g. GLOBAL,TROPICS).")
    p.add_argument("--save-intermediate", dest="save_intermediate", action="store_true", default=None)
    p.add_argument("--logs", dest="enable_logs", action="store_true", default=None)
    p.add_argument(
        "--mpas-grid-file", dest="mpas_grid_file", type=str, default=None,
        help="Path to MPAS grid file (relative to base_dir).",
    )
    p.add_argument(
        "--backend", dest="backend", type=str, default="auto",
        choices=["auto", "mpi", "multiprocessing", "serial"],
        help="Parallel backend: auto (default), mpi, multiprocessing, serial.",
    )
    p.add_argument(
        "--nprocs", dest="nprocs", type=int, default=None,
        help="Number of workers for multiprocessing backend (default: CPU count).",
    )
    p.add_argument(
        "--cache-dir", dest="cache_dir", type=str, default=None,
        help="Shared disk cache for accumulated observations. "
        "Auto-created as a temp dir when using multiprocessing if not specified.",
    )
    p.set_defaults(func=_cmd_run)


def _cmd_run(args: argparse.Namespace) -> None:
    """
    Execute the FSS computation pipeline subcommand handler.
    Resolves the configuration from CLI arguments, applies backend and resolution
    overrides, sets up the shared observation cache directory, and then instantiates
    TaskManager and ParallelProcessor to distribute and run all work-units.

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``run`` subcommand parser.

    Returns:
        None
    """
    import os

    from .parallel import ParallelProcessor
    from .task_manager import TaskManager

    cfg = _resolve_config(args)

    # Parse vxdomain override
    if args.vxdomain is not None:
        cfg = merge_cli_overrides(cfg, {"vxdomain": [d.strip().upper() for d in args.vxdomain.split(",")]})

    # Parse target_resolution to float if numeric
    if args.target_resolution is not None:
        tr = args.target_resolution
        if tr not in ("obs", "fcst"):
            try:
                tr = float(tr)
            except ValueError:
                pass
        cfg = merge_cli_overrides(cfg, {"target_resolution": tr})

    # Parse mpas_grid_file override
    if args.mpas_grid_file is not None:
        cfg = merge_cli_overrides(cfg, {"mpas_grid_file": args.mpas_grid_file})

    # --- Shared observation cache -----------------------------------------
    # Use a deterministic path so that ALL processes (MPI ranks,
    # multiprocessing workers) share the same cache directory.
    # Priority: --cache-dir flag  >  YAML cache_dir  >  auto-derive from
    # output_dir.
    if args.cache_dir:
        cfg = merge_cli_overrides(cfg, {"cache_dir": args.cache_dir})
    elif cfg.cache_dir is None:
        auto_cache = os.path.join(
            cfg.resolve_path(cfg.output_dir), ".obs_cache"
        )
        cfg = merge_cli_overrides(cfg, {"cache_dir": auto_cache})

    tm = TaskManager(cfg)
    pp = ParallelProcessor(
        tm.execute_work_unit,
        backend=args.backend,
        nprocs=args.nprocs,
    )
    units = tm.build_work_units()
    pp.run(units)


# ======================================================================
# Subcommand: extract-csv
# ======================================================================

def _build_extract_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("extract-csv", help="Extract FSS NetCDF results to CSV.")
    _add_common_args(p)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--csv-dir", type=str, default=None)
    p.set_defaults(func=_cmd_extract)


def _cmd_extract(args: argparse.Namespace) -> None:
    """
    Execute the extract-csv subcommand handler.
    Resolves the configuration, instantiates FileManager, and delegates to
    extract_fss_to_csv to scan FSS NetCDF output files and write per-experiment
    CSV summaries.

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``extract-csv`` subcommand parser.

    Returns:
        None
    """
    from .file_manager import FileManager

    cfg = _resolve_config(args)
    _setup_logging(cfg)
    fm = FileManager(cfg)
    fm.extract_fss_to_csv(output_dir=args.output_dir, csv_dir=args.csv_dir)


# ======================================================================
# Subcommand: plot
# ======================================================================

def _build_plot_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("plot", help="Generate metric vs lead-time plots.")
    _add_common_args(p)
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--thresh", type=str, default=None)
    p.add_argument("--window", type=str, default=None)
    p.add_argument("--csv-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--metric", type=str, default=None,
        help="Comma-separated metric names to plot (e.g. fss,pod,csi). "
        "Default: all available metrics in the CSV.",
    )
    p.add_argument("--all", dest="generate_all", action="store_true",
                   help="Generate plots for every (metric, domain, thresh, window) combination.")
    p.set_defaults(func=_cmd_plot)


def _cmd_plot(args: argparse.Namespace) -> None:
    """
    Execute the plot subcommand handler.
    Resolves the configuration, instantiates Visualizer, and generates either a single
    metric-vs-leadtime plot for the specified (domain, threshold, window, metric) combination
    or all combinations when the ``--all`` flag is provided. An optional ``--metric`` filter
    restricts which metrics are plotted.

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``plot`` subcommand parser.

    Returns:
        None
    """
    from .visualizer import Visualizer

    cfg = _resolve_config(args)
    _setup_logging(cfg)
    viz = Visualizer(cfg)

    metrics = None
    if args.metric:
        metrics = [m.strip().lower() for m in args.metric.split(",")]

    if args.generate_all:
        viz.generate_all_plots(csv_dir=args.csv_dir, output_dir=args.output_dir, metrics=metrics)
    elif args.domain and args.thresh and args.window:
        for met in (metrics or ["fss"]):
            viz.plot_fss_vs_leadtime(
                domain=args.domain, thresh=args.thresh, window=args.window,
                csv_dir=args.csv_dir, output_dir=args.output_dir, metric=met,
            )
    else:
        print("Error: provide --domain, --thresh, --window  (or use --all).", file=sys.stderr)
        sys.exit(1)


# ======================================================================
# Subcommand: validate
# ======================================================================

def _build_validate_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("validate", help="List available domains, thresholds, and windows.")
    _add_common_args(p)
    p.add_argument("--csv-dir", type=str, default=None)
    p.set_defaults(func=_cmd_validate)


def _cmd_validate(args: argparse.Namespace) -> None:
    """
    Execute the validate subcommand handler.
    Resolves the configuration, instantiates Visualizer, and prints the unique
    domains, thresholds, and window sizes available in the CSV output directory.
    Exits with code 1 when no CSV data is found.

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``validate`` subcommand parser.

    Returns:
        None
    """
    from .visualizer import Visualizer

    cfg = _resolve_config(args)
    _setup_logging(cfg)
    viz = Visualizer(cfg)
    domains, thresholds, windows = viz.list_available_options(csv_dir=args.csv_dir)

    if domains is None:
        print("No CSV data found.")
        sys.exit(1)

    print(f"Domains:    {', '.join(domains)}")
    print(f"Thresholds: {', '.join(str(t) for t in thresholds)}")  # type: ignore[union-attr]
    print(f"Windows:    {', '.join(str(w) for w in windows)}")  # type: ignore[union-attr]


# ======================================================================
# Config resolution helper
# ======================================================================

def _resolve_config(args: argparse.Namespace) -> ModvxConfig:
    """
    Resolve the active ModvxConfig from a parsed argument namespace.
    If a ``--config`` path was provided, the YAML file is loaded as the base configuration;
    otherwise the default ModvxConfig values are used. Any recognised non-None arguments
    in the namespace are then applied as overrides via merge_cli_overrides. This function
    is called at the start of every subcommand handler to obtain a fully populated
    configuration object ready for use.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments namespace from argparse.

    Returns:
        ModvxConfig: Fully resolved configuration with YAML defaults and CLI overrides merged.
    """
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = ModvxConfig()

    # Collect numeric-override fields
    overrides = {}
    for field_name in (
        "experiment_name",
        "initial_cycle_start",
        "final_cycle_start",
        "forecast_step_hours",
        "observation_interval_hours",
        "cycle_interval_hours",
        "forecast_length_hours",
        "verbose",
        "save_intermediate",
        "enable_logs",
    ):
        val = getattr(args, field_name, None)
        if val is not None:
            overrides[field_name] = val

    if overrides:
        cfg = merge_cli_overrides(cfg, overrides)
    return cfg


# ======================================================================
# Entry-point
# ======================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for the ``modvx`` command-line interface installed via setuptools.
    Constructs the top-level argument parser with four subcommands: ``run``, ``extract-csv``,
    ``plot``, and ``validate``. Each subcommand is registered by its dedicated builder function.
    After parsing, the subcommand's handler is dispatched via the ``func`` default set on
    each subparser.

    Parameters:
        argv (list of str, optional): Argument list to parse; defaults to ``sys.argv[1:]``
            when None.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        prog="modvx",
        description="Model Verification Toolkit — Fraction Skill Score pipeline",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    _build_run_parser(sub)
    _build_extract_parser(sub)
    _build_plot_parser(sub)
    _build_validate_parser(sub)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
