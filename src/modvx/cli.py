#!/usr/bin/env python3

"""
Command-line interface for MODvx.

This module defines the CLI entry point for the modvx package, including subcommands for running the FSS computation pipeline, extracting results to CSV, generating plots, and validating available options. It uses argparse to construct a user-friendly interface with shared configuration loading and override logic. Each subcommand has its own handler function that resolves the configuration, sets up logging, and delegates to the appropriate components of the MODvx package.

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
    This helper configures the root logging settings based on the provided ModvxConfig. It sets the log level to DEBUG if verbose mode is enabled, otherwise INFO. The log format includes timestamps, log levels, logger names, and messages for consistent and informative logging across all subcommands and modules. This function is called at the start of each subcommand handler to ensure that logging is configured according to the resolved configuration. 

    Parameters:
        config (ModvxConfig): Run configuration used to determine the logging level.

    Returns:
        None
    """
    # Set log level to DEBUG if verbose mode is enabled, otherwise INFO.
    level = logging.DEBUG if config.verbose else logging.INFO

    # Configure root logging with a consistent format and the determined log level.
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def add_shared_cli_args(parser: argparse.ArgumentParser) -> None:
    """
    This helper registers CLI arguments that are common across all subcommands on the provided argument parser. It adds a ``--config`` argument for specifying a YAML configuration file and a ``--verbose`` flag to enable verbose logging. These shared arguments allow users to control the configuration source and logging verbosity consistently across different subcommands without needing to redefine them in each subparser. 

    Parameters:
        parser (argparse.ArgumentParser): Argument parser instance for a specific subcommand.

    Returns:
        None
    """
    # Common arguments for all subcommands
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: use built-in defaults).",
    )

    # If verbose is specified, set it to True; otherwise, leave it as None 
    parser.add_argument("--verbose", action="store_true", default=None)


def build_run_subparser(sub: argparse._SubParsersAction) -> None:
    """
    This helper registers all CLI arguments for the ``run`` subcommand on the provided subparsers action. It creates and configures the ``run`` subparser with arguments for experiment naming, cycle and forecast period specification, resolution overrides, backend selection for parallel processing, and control flags for saving intermediate fields and enabling logs. Common arguments such as ``--config`` and ``--verbose`` are added via ``add_shared_cli_args``. The handler function for this subcommand is set to ``handle_run_subcommand``. 

    Parameters:
        sub (argparse._SubParsersAction): The subparsers action to register the 'run' subcommand on.

    Returns:
        None
    """
    # Create the 'run' subparser and add shared arguments
    p = sub.add_parser("run", help="Execute the FSS computation pipeline.")

    # Add shared arguments like --config and --verbose
    add_shared_cli_args(p)

    # Experiment name for labeling outputs and logs
    p.add_argument("--expname", dest="experiment_name", type=str, default=None)

    # Specify the start of the verification period formatted as a datetime string 
    p.add_argument("--start", dest="initial_cycle_start", type=parse_datetime_string, default=None)

    # Specify the end of the verification period formatted as a datetime string 
    p.add_argument("--end", dest="final_cycle_start", type=parse_datetime_string, default=None)

    # Set the forecast step in hours (e.g., 6 for 6-hourly forecasts)
    p.add_argument("--forecast-step", dest="forecast_step_hours", type=int, default=None)

    # Set the observation interval in hours (e.g., 1 for hourly observations)
    p.add_argument("--obs-interval", dest="observation_interval_hours", type=int, default=None)

    # Set the cycle interval in hours (e.g., 24) 
    p.add_argument("--cycle-interval", dest="cycle_interval_hours", type=int, default=None)

    # Set the forecast length in hours (e.g., 48) for each cycle.
    p.add_argument("--forecast-length", dest="forecast_length_hours", type=int, default=None)

    # Set the precipitation accumulation period in hours (e.g., 3 for 3h accumulated precip)
    p.add_argument("--precip-accum", dest="precip_accum_hours", type=int, default=None,
                   help="Precipitation accumulation period in hours (e.g. 3 for 3h accum). "
                   "Default: same as forecast-step.")

    # Set the target resolution to remap MPAS forecasts
    p.add_argument("--target-resolution", dest="target_resolution", type=str, default=None)

    # Specify verification domains as a comma-separated string (e.g. "GLOBAL,TROPICS") 
    p.add_argument("--vxdomain", dest="vxdomain", type=str, default=None,
                   help="Comma-separated verification domains (e.g. GLOBAL,TROPICS).")
    
    # Control whether intermediate fields are saved to disk with an explicit flag.
    p.add_argument("--save-intermediate", dest="save_intermediate", action="store_true", default=None)

    # Control logging with an explicit flag to enable logs
    p.add_argument("--logs", dest="enable_logs", action="store_true", default=None)

    # Specify the required MPAS grid file path if using native MPAS unstructured data. 
    p.add_argument(
        "--mpas-grid-file", dest="mpas_grid_file", type=str, default=None,
        help="Path to MPAS grid file (relative to base_dir).",
    )

    # Parallel backend selection for the run subcommand, with choices and a default of 'auto'.
    p.add_argument(
        "--backend", dest="backend", type=str, default="auto",
        choices=["auto", "mpi", "multiprocessing", "serial"],
        help="Parallel backend: auto (default), mpi, multiprocessing, serial.",
    )

    # Specify the number of worker processes for the multiprocessing backend.
    p.add_argument(
        "--nprocs", dest="nprocs", type=int, default=None,
        help="Number of workers for multiprocessing backend (default: CPU count).",
    )

    # Specify the shared observation cache directory for parallel workers. 
    p.add_argument(
        "--cache-dir", dest="cache_dir", type=str, default=None,
        help="Shared disk cache for accumulated observations. "
        "Auto-created as a temp dir when using multiprocessing if not specified.",
    )

    # Set the default function to handle this subcommand when invoked
    p.set_defaults(func=handle_run_subcommand)


def parse_vxdomain_tokens(raw: str) -> list[str]:
    """
    This helper parses the raw comma-separated string of verification domains provided via the CLI into a list of normalized, uppercase domain tokens. For example, an input of "GLOBAL,TROPICS" would be parsed into ["GLOBAL", "TROPICS"]. This allows users to specify multiple verification domains in a single argument while ensuring consistent formatting for downstream processing. 

    Parameters:
        raw (str): Comma-separated domain names (e.g., "GLOBAL,TROPICS").

    Returns:
        list[str]: Normalised, uppercase domain tokens.
    """
    # Extract domain names by splitting on commas
    return [d.strip().upper() for d in raw.split(",")]


def parse_target_resolution(raw: str) -> float | str:
    """
    This helper attempts to parse the raw target resolution string provided via the CLI into a numeric value in degrees. If the input is a special token like "obs" or "fcst", it returns the token as-is without parsing. If the input can be successfully parsed as a float, it returns the numeric value; otherwise, it returns the original string token. This allows users to specify either a numeric resolution (e.g., "0.25") or special tokens that indicate using the native resolution of observations or forecasts. 

    Parameters:
        raw (str): Raw resolution value from CLI or configuration.

    Returns:
        float | str: Numeric resolution in degrees when parseable, otherwise the original string token.
    """
    # If the input is a special token like "obs" or "fcst", return it as-is without parsing.
    if raw in ("obs", "fcst"):
        return raw
    
    try:
        # If parsing succeeds, return the numeric value as a float.
        return float(raw)
    except ValueError:
        # If parsing fails, return the original string token for downstream handling.
        return raw


def resolve_observation_cache_dir(config: ModvxConfig, 
                                  cli_cache_dir: str | None) -> str | None:
    """
    This helper determines the appropriate shared observation cache directory to use for parallel processing based on the provided configuration and CLI arguments. The logic prioritizes the CLI argument if provided, then checks the YAML configuration, and finally falls back to a default path within the output directory if neither is specified. This ensures that all parallel workers have a consistent location for caching accumulated observations, which is essential for efficient execution of the FSS computation pipeline when using multiprocessing or MPI backends. 

    Parameters:
        config (ModvxConfig): The resolved configuration object, potentially with YAML values.
        cli_cache_dir (str | None): The value of the ``--cache-dir`` CLI argument, which may be None if not provided.

    Returns:
        str | None: The resolved cache directory path to use, or None if no override is needed.
    """
    import os

    # CLI flag takes precedence over YAML config; if provided, use it directly.
    if cli_cache_dir:
        return cli_cache_dir
    
    # If the CLI flag was not used and the YAML config does not have a cache_dir
    if config.cache_dir is None:
        return os.path.join(config.resolve_relative_path(config.output_dir), ".obs_cache")
    
    # No override needed; the YAML config already has a cache_dir set and the CLI flag was not used.
    return None


def handle_run_subcommand(args: argparse.Namespace) -> None:
    """
    This function executes the logic for the ``run`` subcommand. It resolves the configuration by loading from a YAML file if specified and applying any CLI overrides. It then initializes the TaskManager with the resolved configuration to build the set of work units for execution. A ParallelProcessor is instantiated with the TaskManager's execute_work_unit method and the selected backend. Finally, all work units are run in parallel using the ParallelProcessor. This function serves as the main entry point for executing the FSS computation pipeline when users invoke the ``modvx run`` command. 

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``run`` subcommand parser.

    Returns:
        None
    """
    from .parallel import ParallelProcessor
    from .task_manager import TaskManager

    # Resolve the base configuration from the provided CLI arguments
    config = resolve_config_from_namespace(args)

    # Initialize a dictionary to hold any CLI overrides that need to be applied to the configuration before execution.
    overrides: dict = {}

    # Apply CLI overrides for verification domains if provided
    if args.vxdomain is not None:
        overrides["vxdomain"] = parse_vxdomain_tokens(args.vxdomain)

    # Apply CLI override for target resolution if provided
    if args.target_resolution is not None:
        overrides["target_resolution"] = parse_target_resolution(args.target_resolution)

    # Apply CLI override for MPAS grid file if provided
    if args.mpas_grid_file is not None:
        overrides["mpas_grid_file"] = args.mpas_grid_file

    # Determine the shared observation cache directory based on CLI and config values
    cache_dir = resolve_observation_cache_dir(config, args.cache_dir)

    # Override the cache_dir in the configuration if a new path was resolved
    if cache_dir is not None:
        overrides["cache_dir"] = cache_dir

    # Apply any collected CLI overrides to the configuration before proceeding with task execution.
    if overrides:
        config = apply_cli_overrides(config, overrides)

    # Initialize the TaskManager with the resolved configuration to build the set of work units for execution.
    tm = TaskManager(config)

    # Initialize the ParallelProcessor with the TaskManager's execute_work_unit method and the selected backend.
    pp = ParallelProcessor(tm.execute_work_unit, backend=args.backend, nprocs=args.nprocs)

    # Run all work units in parallel using the selected backend. 
    pp.run(tm.build_work_units())


def build_extract_subparser(sub: argparse._SubParsersAction) -> None:
    """
    This helper registers all CLI arguments for the ``extract-csv`` subcommand on the provided subparsers action. It creates and configures the ``extract-csv`` subparser with a single ``--output-dir`` argument for specifying where to look for FSS NetCDF output files and a ``--csv-dir`` argument for controlling where to write the extracted CSV summaries. Common arguments such as ``--config`` and ``--verbose`` are added via ``add_shared_cli_args``. The handler function for this subcommand is set to ``handle_extract_subcommand``. 

    Parameters:
        sub (argparse._SubParsersAction): The subparsers action to register the 'extract-csv' subcommand on.

    Returns:
        None
    """
    # Create the 'extract-csv' subparser and add shared arguments
    p = sub.add_parser("extract-csv", help="Extract FSS NetCDF results to CSV.")

    # Add shared arguments like --config and --verbose
    add_shared_cli_args(p)

    # Set the default function to handle this subcommand when invoked
    p.add_argument("--output-dir", type=str, default=None, help="Directory containing FSS NetCDF output files.")

    # Set the default function to handle this subcommand when invoked
    p.add_argument("--csv-dir", type=str, default=None, help="Directory to write extracted CSV summaries.")

    # Set the default function to handle this subcommand when invoked
    p.set_defaults(func=handle_extract_subcommand)


def handle_extract_subcommand(args: argparse.Namespace) -> None:
    """
    This function executes the logic for the ``extract-csv`` subcommand. It resolves the configuration by loading from a YAML file if specified and applying any CLI overrides. It then instantiates the FileManager with the resolved configuration and calls its method to extract FSS results from NetCDF files in the specified output directory, writing CSV summaries to the specified CSV directory. This function serves as the main entry point for users to generate CSV summaries of FSS results after running the main pipeline when they invoke the ``modvx extract-csv`` command. 

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``extract-csv`` subcommand parser.

    Returns:
        None
    """
    from .file_manager import FileManager

    # Resolve the base configuration from the provided CLI arguments
    config = resolve_config_from_namespace(args)

    # Configure root logging based on the resolved configuration 
    configure_root_logging(config)

    # Instantiate the FileManager with the resolved configuration 
    fm = FileManager(config)

    # Extract FSS results from NetCDF files in the output directory and write CSV summaries
    fm.extract_fss_to_csv(output_dir=args.output_dir, csv_dir=args.csv_dir)


def build_plot_subparser(sub: argparse._SubParsersAction) -> None:
    """
    This helper registers all CLI arguments for the ``plot`` subcommand on the provided subparsers action. It creates and configures the ``plot`` subparser with arguments for specifying the verification domain, percentile threshold, accumulation window size, and an optional comma-separated list of metrics to plot. A flag is also provided to generate plots for all combinations of available options. Common arguments such as ``--config`` and ``--verbose`` are added via ``add_shared_cli_args``. The handler function for this subcommand is set to ``handle_plot_subcommand``.

    Parameters:
        sub (argparse._SubParsersAction): The subparsers action to register the 'plot' subcommand on.

    Returns:
        None
    """
    # Create the 'plot' subparser and add shared arguments
    p = sub.add_parser("plot", help="Generate metric vs lead-time plots.")

    # Add shared arguments like --config and --verbose
    add_shared_cli_args(p)

    # Specify the verification domain to plot (e.g., "GLOBAL", "TROPICS"). 
    p.add_argument("--domain", type=str, default=None, help="Verification domain to plot (e.g., 'GLOBAL', 'TROPICS').")

    # Specify the percentile threshold as a string (e.g., "0.1", "0.5", "obs")
    p.add_argument("--thresh", type=str, default=None, help="Percentile threshold to plot (e.g., '0.1', '0.5', 'obs').")

    # Specify the accumulation window size as a string (e.g., "1h", "3h", "6h") 
    p.add_argument("--window", type=str, default=None, help="Accumulation window size to plot (e.g., '1h', '3h', '6h').")

    # Set the default function to handle this subcommand when invoked
    p.add_argument("--csv-dir", type=str, default=None, help="Directory containing CSV summaries of FSS results.")

    # Set the default function to handle this subcommand when invoked
    p.add_argument("--output-dir", type=str, default=None, help="Directory to write generated plots.")

    # Optionally, specify a comma-separated list of metrics to plot (e.g., "fss,pod,csi"). 
    p.add_argument(
        "--metric", type=str, default=None,
        help="Comma-separated metric names to plot (e.g. fss,pod,csi). "
        "Default: all available metrics in the CSV.",
    )

    # Generate plots for all combinations when the --all flag is provided
    p.add_argument("--all", dest="generate_all", action="store_true",
                   help="Generate plots for every (metric, domain, thresh, window) combination.")

    # Set the default function to handle this subcommand when invoked
    p.set_defaults(func=handle_plot_subcommand)


def handle_plot_subcommand(args: argparse.Namespace) -> None:
    """
    This function executes the logic for the ``plot`` subcommand. It resolves the configuration by loading from a YAML file if specified and applying any CLI overrides. It then instantiates the Visualizer with the resolved configuration. If specific metric, domain, threshold, and window arguments are provided, it generates a single plot for that combination; if the --all flag is set, it generates plots for all available combinations in the CSV directory. If required arguments are missing for the single-plot case, it prints an error message and exits with code 1. This function serves as the main entry point for users to generate visualizations of FSS results when they invoke the ``modvx plot`` command. 

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``plot`` subcommand parser.

    Returns:
        None
    """
    from .visualizer import Visualizer

    # Resolve the base configuration from the provided CLI arguments
    config = resolve_config_from_namespace(args)

    # Configure root logging based on the resolved configuration
    configure_root_logging(config)

    # Instantiate the Visualizer with the resolved configuration 
    viz = Visualizer(config)

    # Initialize the metrics variable to None
    metrics = None

    # If the --metric argument is provided, parse it into a list of metric names to plot. 
    if args.metric:
        metrics = [m.strip().lower() for m in args.metric.split(",")]

    if args.generate_all:
        # Generate plots for all combinations of metrics, domains, thresholds, and windows available
        viz.generate_all_plots(csv_dir=args.csv_dir, output_dir=args.output_dir, metrics=metrics)
    elif args.domain and args.thresh and args.window:
        for met in (metrics or ["fss"]):
            # Generate a single plot for the specified metric, domain, threshold, and window combination.
            viz.plot_fss_vs_leadtime(
                domain=args.domain, thresh=args.thresh, window=args.window,
                csv_dir=args.csv_dir, output_dir=args.output_dir, metric=met,
            )
    else:
        # If required arguments are missing for the single-plot case, print an error message and exit with code 1.
        print("Error: provide --domain, --thresh, --window  (or use --all).", file=sys.stderr)
        sys.exit(1)


def build_validate_subparser(sub: argparse._SubParsersAction) -> None:
    """
    This helper registers all CLI arguments for the ``validate`` subcommand on the provided subparsers action. It creates and configures the ``validate`` subparser with a single optional argument for specifying the CSV directory to validate. Common arguments such as ``--config`` and ``--verbose`` are added via ``add_shared_cli_args``. The handler function for this subcommand is set to ``handle_validate_subcommand``. 

    Parameters:
        sub (argparse._SubParsersAction): The subparsers action to register the 'validate' subcommand on.

    Returns:
        None
    """
    # Create the 'validate' subparser and add shared arguments
    p = sub.add_parser("validate", help="List available domains, thresholds, and windows.")

    # Add shared arguments like --config and --verbose
    add_shared_cli_args(p)

    # Set the default function to handle this subcommand when invoked
    p.add_argument("--csv-dir", type=str, default=None, help="Directory containing CSV summaries of FSS results.")

    # Set the default function to handle this subcommand when invoked
    p.set_defaults(func=handle_validate_subcommand)


def handle_validate_subcommand(args: argparse.Namespace) -> None:
    """
    This function executes the logic for the ``validate`` subcommand. It resolves the configuration by loading from a YAML file if specified and applying any CLI overrides. It then instantiates the Visualizer with the resolved configuration and calls its method to list the unique verification domains, percentile thresholds, and accumulation window sizes available in the specified CSV directory. If no CSV data is found, it prints a message and exits with code 1. Otherwise, it prints the available options in a user-friendly format. This function serves as a utility for users to quickly check what combinations of parameters are available in their CSV summaries before attempting to generate plots or run analyses. 

    Parameters:
        args (argparse.Namespace): Parsed arguments from the ``validate`` subcommand parser.

    Returns:
        None
    """
    from .visualizer import Visualizer

    # Resolve the base configuration from the provided CLI arguments
    config = resolve_config_from_namespace(args)

    # Configure root logging based on the resolved configuration
    configure_root_logging(config)

    # Instantiate the Visualizer with the resolved configuration
    viz = Visualizer(config)

    # List the unique domains, thresholds, and window sizes available in the CSV output directory.
    domains, thresholds, windows = viz.list_available_options(csv_dir=args.csv_dir)

    # If no domains are found, print a message and exit with code 1 
    if domains is None:
        print("No CSV data found.")
        sys.exit(1)

    # Print the available domains, thresholds, and windows in a user-friendly format.
    print(f"Domains:    {', '.join(domains)}")
    print(f"Thresholds: {', '.join(str(t) for t in thresholds)}")  # type: ignore[union-attr]
    print(f"Windows:    {', '.join(str(w) for w in windows)}")  # type: ignore[union-attr]


def resolve_config_from_namespace(args: argparse.Namespace) -> ModvxConfig:
    """
    This helper function resolves the ModvxConfig configuration object based on the provided argparse.Namespace from the CLI. It first checks if a YAML configuration file is specified via the ``--config`` argument and loads it if present; otherwise, it starts with the default configuration values. It then iterates through potential CLI override fields, collecting any that are provided into an overrides dictionary. Finally, it applies any collected CLI overrides to the base configuration using the ``apply_cli_overrides`` function and returns the fully resolved ModvxConfig object. This function centralizes the logic for merging YAML defaults with CLI overrides, ensuring consistent configuration resolution across all subcommands. 

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments namespace from argparse.

    Returns:
        ModvxConfig: Fully resolved configuration with YAML defaults and CLI overrides merged.
    """
    if args.config:
        # Load the base configuration from the specified YAML file if the --config argument is provided
        cfg = load_config_from_yaml(args.config)
    else:
        # Otherwise, start with the default configuration values
        cfg = ModvxConfig()

    # Initialize an empty dictionary to collect CLI overrides
    overrides = {}

    # Check for each potential override field in the args namespace and add it to the overrides dict 
    for field_name in (
        "experiment_name",
        "initial_cycle_start",
        "final_cycle_start",
        "forecast_step_hours",
        "observation_interval_hours",
        "cycle_interval_hours",
        "forecast_length_hours",
        "precip_accum_hours",
        "verbose",
        "save_intermediate",
        "enable_logs",
    ):
        val = getattr(args, field_name, None)
        if val is not None:
            overrides[field_name] = val

    # Apply any collected CLI overrides to the configuration
    if overrides:
        cfg = apply_cli_overrides(cfg, overrides)

    # Return the resolved configuration object 
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    """
    This is the main entry point for the modvx command-line interface. It constructs the argument parser with subcommands for running the FSS computation pipeline, extracting results to CSV, generating plots, and validating available options. It parses the command-line arguments and dispatches to the appropriate handler function based on the selected subcommand. This function is invoked when users run the ``modvx`` command from the terminal, allowing them to interact with the various functionalities of the MODvx package through a user-friendly CLI. 

    Parameters:
        argv (list of str, optional): Argument list to parse; defaults to ``sys.argv[1:]`` when None.

    Returns:
        None
    """
    # Capture the command-line arguments
    parser = argparse.ArgumentParser(
        prog="modvx",
        description="Model Verification Toolkit for MPAS Precipitation Forecasts",
    )

    # Create subparsers for the different commands 
    sub = parser.add_subparsers(dest="command")

    # Make the subcommand required 
    sub.required = True

    # Build the run subparser for executing the metric computation pipeline.
    build_run_subparser(sub)

    # Build the extract-csv subparser for extracting FSS results from NetCDF files into CSV summaries.
    build_extract_subparser(sub)

    # Build the plot subparser for generating metric-vs-leadtime plots from CSV summaries.
    build_plot_subparser(sub)

    # Build the validate subparser for listing available domains, thresholds, and windows in the CSV output directory.
    build_validate_subparser(sub)

    # Parse the command-line arguments
    args = parser.parse_args(argv)

    # Dispatch to the appropriate subcommand handler 
    args.func(args)


# Invoke the main function when this script is executed directly
if __name__ == "__main__":
    main()
