#!/usr/bin/env python3

"""
Command-line interface for modvx.

This module defines the main entry point for the modvx verification toolkit when invoked from the command line. It handles argument parsing, configuration loading, and orchestrates the execution of the verification workflow by instantiating the TaskManager and ParallelProcessor. The CLI provides a user-friendly interface for running verification experiments with custom configurations and supports options for logging, output management, and parallel execution settings. By centralizing the CLI logic in this module, we can ensure a consistent user experience and provide a clear starting point for users to interact with the modvx toolkit.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from __future__ import annotations

import sys
import logging
import argparse
from typing import List, Optional

from .utils import parse_datetime_string
from .config import ModvxConfig, load_config_from_yaml, apply_cli_overrides


def configure_root_logging(config: ModvxConfig) -> None:
    """
    Configure root logging so that logger.info messages are visible on the console. The ``run`` subcommand relies on TaskManager to do this; other subcommands bypass TaskManager entirely, so this helper fills the gap. Sets the log level to DEBUG when verbose mode is enabled in the configuration, or INFO otherwise. Uses ``logging.basicConfig`` with ``force=True`` so the handler is always applied, even if a root handler was already installed.

    Parameters:
        config (ModvxConfig): Run configuration used to determine the logging level.

    Returns:
        None
    """
    level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

# Backwards-compatible alias
# _setup_logging = configure_root_logging


def add_shared_cli_args(parser: argparse.ArgumentParser) -> None:
    """
    Register arguments that are common to all modvx subcommands on a given parser. Currently registers the ``--config`` flag for pointing to a YAML configuration file and ``--verbose`` for enabling detailed debug logging. Centralising these shared arguments avoids repetition across _build_run_parser, _build_extract_parser, _build_plot_parser, and _build_validate_parser.

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

# Backwards-compatible alias
# Note: legacy short-name alias removed; use `add_shared_cli_args`.


# ======================================================================
# Subcommand: run
# ======================================================================

def build_run_subparser(sub: argparse._SubParsersAction) -> None:
    """
    Register all CLI arguments for the ``run`` subcommand on the provided subparsers action. Creates and configures the ``run`` subparser with arguments for experiment name, cycle start/end, forecast and observation timing, verification domains, target resolution, and the MPAS grid file path. Also registers backend selection (``--backend``), worker count (``--nprocs``), and a shared observation cache directory (``--cache-dir``) for multiprocessing runs.

    Parameters:
        sub (argparse._SubParsersAction): The subparsers action to register the 'run' subcommand on.

    Returns:
        None
    """
    p = sub.add_parser("run", help="Execute the FSS computation pipeline.")
    add_shared_cli_args(p)

    p.add_argument("--expname", dest="experiment_name", type=str, default=None)
    p.add_argument("--start", dest="initial_cycle_start", type=parse_datetime_string, default=None)
    p.add_argument("--end", dest="final_cycle_start", type=parse_datetime_string, default=None)
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
    p.set_defaults(func=handle_run_subcommand)

# Note: legacy short-name alias removed; use `build_run_subparser`.


def parse_vxdomain_tokens(raw: str) -> list[str]:
    """
    This helper parses a comma-separated string of verification domain names into a list of uppercase tokens. For example, the input "GLOBAL,TROPICS" would be transformed into ["GLOBAL", "TROPICS"]. The parsing is case-insensitive and trims whitespace around domain names. This allows users to specify multiple verification domains in a single CLI argument while ensuring consistent formatting for downstream processing.

    Parameters:
        raw (str): Comma-separated domain names (e.g., "GLOBAL,TROPICS").

    Returns:
        list[str]: Normalised, uppercase domain tokens.
    """
    return [d.strip().upper() for d in raw.split(",")]

# Backwards-compatible alias
# _parse_vxdomain = parse_vxdomain_tokens


def parse_target_resolution(raw: str) -> float | str:
    """
    This helper attempts to parse a target resolution value from the CLI. If the input string can be converted to a float, that numeric value is returned (e.g., "0.25" → 0.25). If the input is not parseable as a float (e.g., "obs", "fcst"), the original string is returned unchanged. This allows for flexible specification of either numeric resolutions or special tokens in the same argument.

    Parameters:
        raw (str): Raw resolution value from CLI or configuration.

    Returns:
        float | str: Numeric resolution in degrees when parseable, otherwise the original string token.
    """
    if raw in ("obs", "fcst"):
        return raw
    try:
        return float(raw)
    except ValueError:
        return raw

# Note: legacy short-name alias removed; use `parse_target_resolution`.


def resolve_observation_cache_dir(config: ModvxConfig, cli_cache_dir: str | None) -> str | None:
    """
    This helper determines the shared observation cache directory based on a priority order: the CLI ``--cache-dir`` flag takes precedence, followed by the YAML configuration's ``cache_dir`` value. If neither is provided, it auto-derives a cache directory path by appending ".obs_cache" to the resolved output directory. If the YAML config already has a non-None ``cache_dir`` and the CLI flag was not used, this function returns None to indicate that no override is needed. The returned path is used to ensure that all parallel workers read and write from the same location when caching accumulated observations.

    Parameters:
        config (ModvxConfig): The resolved configuration object, potentially with YAML values.
        cli_cache_dir (str | None): The value of the ``--cache-dir`` CLI argument, which may be None if not provided.

    Returns:
        str | None: The resolved cache directory path to use, or None if no override is needed.
    """
    import os

    if cli_cache_dir:
        return cli_cache_dir
    if config.cache_dir is None:
        return os.path.join(config.resolve_relative_path(config.output_dir), ".obs_cache")
    return None



def handle_run_subcommand(args: argparse.Namespace) -> None:
    """
    Execute the FSS computation pipeline subcommand handler. Resolves the configuration from CLI arguments, applies backend and resolution overrides, sets up the shared observation cache directory, and then instantiates TaskManager and ParallelProcessor to distribute and run all work-units.

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``run`` subcommand parser.

    Returns:
        None
    """
    from .parallel import ParallelProcessor
    from .task_manager import TaskManager

    config = resolve_config_from_namespace(args)

    # Collect all run-specific overrides and apply in one shot.
    overrides: dict = {}
    if args.vxdomain is not None:
        overrides["vxdomain"] = parse_vxdomain_tokens(args.vxdomain)
    if args.target_resolution is not None:
        overrides["target_resolution"] = parse_target_resolution(args.target_resolution)
    if args.mpas_grid_file is not None:
        overrides["mpas_grid_file"] = args.mpas_grid_file
    cache_dir = resolve_observation_cache_dir(config, args.cache_dir)
    if cache_dir is not None:
        overrides["cache_dir"] = cache_dir
    if overrides:
        config = apply_cli_overrides(config, overrides)
    tm = TaskManager(config)
    pp = ParallelProcessor(tm.execute_work_unit, backend=args.backend, nprocs=args.nprocs)
    pp.run(tm.build_work_units())

# Note: legacy short-name alias removed; use `handle_run_subcommand`.


# ======================================================================
# Subcommand: extract-csv
# ======================================================================

def build_extract_subparser(sub: argparse._SubParsersAction) -> None:
    """
    Register all CLI arguments for the ``extract-csv`` subcommand on the provided subparsers action. Creates and configures the ``extract-csv`` subparser with arguments for locating FSS NetCDF output files (``--output-dir``) and writing per-experiment CSV summaries (``--csv-dir``). Common arguments such as ``--config`` and ``--verbose`` are added via ``_add_common_args``.

    Parameters:
        sub (argparse._SubParsersAction): The subparsers action to register the 'extract-csv' subcommand on.

    Returns:
        None
    """
    p = sub.add_parser("extract-csv", help="Extract FSS NetCDF results to CSV.")
    add_shared_cli_args(p)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--csv-dir", type=str, default=None)
    p.set_defaults(func=handle_extract_subcommand)

# Note: legacy short-name alias removed; use `build_extract_subparser`.


def handle_extract_subcommand(args: argparse.Namespace) -> None:
    """
    Execute the extract-csv subcommand handler. Resolves the configuration, instantiates FileManager, and delegates to extract_fss_to_csv to scan FSS NetCDF output files and write per-experiment CSV summaries.

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``extract-csv`` subcommand parser.

    Returns:
        None
    """
    from .file_manager import FileManager

    config = resolve_config_from_namespace(args)
    configure_root_logging(config)
    fm = FileManager(config)
    fm.extract_fss_to_csv(output_dir=args.output_dir, csv_dir=args.csv_dir)

# Note: legacy short-name alias removed; use `handle_extract_subcommand`.


# ======================================================================
# Subcommand: plot
# ======================================================================

def build_plot_subparser(sub: argparse._SubParsersAction) -> None:
    """
    Register all CLI arguments for the ``plot`` subcommand on the provided subparsers action. Creates and configures the ``plot`` subparser with arguments for filtering by domain, threshold, and accumulation window, and for controlling output locations (``--csv-dir``, ``--output-dir``). Supports an optional ``--metric`` filter for selecting specific metrics to plot and an ``--all`` flag to generate every available combination automatically.

    Parameters:
        sub (argparse._SubParsersAction): The subparsers action to register the 'plot' subcommand on.

    Returns:
        None
    """
    p = sub.add_parser("plot", help="Generate metric vs lead-time plots.")
    add_shared_cli_args(p)
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
    p.set_defaults(func=handle_plot_subcommand)

# Note: legacy short-name alias removed; use `build_plot_subparser`.


def handle_plot_subcommand(args: argparse.Namespace) -> None:
    """
    Execute the plot subcommand handler. Resolves the configuration, instantiates Visualizer, and generates either a single metric-vs-leadtime plot for the specified (domain, threshold, window, metric) combination or all combinations when the ``--all`` flag is provided. An optional ``--metric`` filter restricts which metrics are plotted.

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``plot`` subcommand parser.

    Returns:
        None
    """
    from .visualizer import Visualizer

    config = resolve_config_from_namespace(args)
    configure_root_logging(config)
    viz = Visualizer(config)

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

    # Note: legacy short-name alias removed; use `handle_plot_subcommand`.


# ======================================================================
# Subcommand: validate
# ======================================================================

def build_validate_subparser(sub: argparse._SubParsersAction) -> None:
    """
    Register all CLI arguments for the ``validate`` subcommand on the provided subparsers action. Creates and configures the ``validate`` subparser with a single ``--csv-dir`` argument for specifying where to look for CSV output files. The subcommand prints the unique domains, thresholds, and window sizes available and exits with code 1 when no data is found.

    Parameters:
        sub (argparse._SubParsersAction): The subparsers action to register the 'validate' subcommand on.

    Returns:
        None
    """
    p = sub.add_parser("validate", help="List available domains, thresholds, and windows.")
    add_shared_cli_args(p)
    p.add_argument("--csv-dir", type=str, default=None)
    p.set_defaults(func=handle_validate_subcommand)

# Note: legacy short-name alias removed; use `build_validate_subparser`.


def handle_validate_subcommand(args: argparse.Namespace) -> None:
    """
    Execute the validate subcommand handler. Resolves the configuration, instantiates Visualizer, and prints the unique domains, thresholds, and window sizes available in the CSV output directory. Exits with code 1 when no CSV data is found.

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``validate`` subcommand parser.

    Returns:
        None
    """
    from .visualizer import Visualizer

    config = resolve_config_from_namespace(args)
    configure_root_logging(config)
    viz = Visualizer(config)
    domains, thresholds, windows = viz.list_available_options(csv_dir=args.csv_dir)

    if domains is None:
        print("No CSV data found.")
        sys.exit(1)

    print(f"Domains:    {', '.join(domains)}")
    print(f"Thresholds: {', '.join(str(t) for t in thresholds)}")  # type: ignore[union-attr]
    print(f"Windows:    {', '.join(str(w) for w in windows)}")  # type: ignore[union-attr]

# Note: legacy short-name alias removed; use `handle_validate_subcommand`.


# ======================================================================
# Config resolution helper
# ======================================================================

def resolve_config_from_namespace(args: argparse.Namespace) -> ModvxConfig:
    """
    Resolve the active ModvxConfig from a parsed argument namespace. If a ``--config`` path was provided, the YAML file is loaded as the base configuration; otherwise the default ModvxConfig values are used. Any recognised non-None arguments in the namespace are then applied as overrides via merge_cli_overrides. This function is called at the start of every subcommand handler to obtain a fully populated configuration object ready for use.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments namespace from argparse.

    Returns:
        ModvxConfig: Fully resolved configuration with YAML defaults and CLI overrides merged.
    """
    if args.config:
        cfg = load_config_from_yaml(args.config)
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
        cfg = apply_cli_overrides(cfg, overrides)
    return cfg



# ======================================================================
# Entry-point
# ======================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for the ``modvx`` command-line interface installed via setuptools. Constructs the top-level argument parser with four subcommands: ``run``, ``extract-csv``, ``plot``, and ``validate``. Each subcommand is registered by its dedicated builder function. After parsing, the subcommand's handler is dispatched via the ``func`` default set on each subparser.

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

    build_run_subparser(sub)
    build_extract_subparser(sub)
    build_plot_subparser(sub)
    build_validate_subparser(sub)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
