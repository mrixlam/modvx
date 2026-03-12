#!/usr/bin/env python3

"""
Task orchestration for modvx.

This module defines the TaskManager class, which is responsible for constructing and managing the set of verification tasks to be executed based on the provided configuration. The TaskManager generates a comprehensive list of work units that represent individual combinations of forecast cycles, valid times, domains, thresholds, and window sizes as specified in the config. It also provides methods to retrieve these tasks in a format suitable for parallel processing. By centralizing task construction logic, the TaskManager ensures that all downstream components receive a consistent and complete set of work units to process.
 
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from __future__ import annotations

import os
import logging
import datetime
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .config import ModvxConfig
from .file_manager import FileManager
from .perf_metrics import PerfMetrics
from .data_validator import DataValidator
from .utils import iterate_forecast_cycle_starts


logger = logging.getLogger(__name__)


class TaskManager:
    """
    Orchestrate the end-to-end FSS verification pipeline for a configured experiment. On construction, TaskManager instantiates FileManager, DataValidator, and PerfMetrics and configures Python logging according to the config flags. It exposes two key methods: build_work_units generates the (cycle × region) task list, and execute_work_unit processes a single task by loading data, preparing fields, computing FSS across all parameter combinations, and persisting results.

    Parameters:
        config (ModvxConfig): Fully resolved run configuration with all experiment settings.
    """

    def __init__(self, config: ModvxConfig) -> None:
        """
        This constructor initializes the TaskManager with the provided configuration, setting up necessary components and logging. It creates instances of FileManager, DataValidator, and PerfMetrics using the config, and calls a helper method to configure logging based on the config settings. The TaskManager is then ready to build work units and execute tasks as defined by the configuration.

        Parameters:
            config (ModvxConfig): Fully resolved run configuration with all experiment settings.

        Returns:
            None
        """
        # Store the provided configuration for use in task construction and execution.
        self.config = config

        # Initialize the file manager with the provided configuration
        self.file_manager = FileManager(config)

        # Initialize the data validator with the provided configuration 
        self.data_validator = DataValidator(config)

        # Initialize the performance metrics calculator with the provided configuration
        self.perf_metrics = PerfMetrics(config)

        # Configure logging based on the config settings to ensure that all components use a consistent logging setup. 
        self._setup_logging()


    def _setup_logging(self) -> None:
        """
        Configure Python's root logging system based on the current ModvxConfig settings. Log level is set to DEBUG when verbose mode is enabled, and INFO otherwise. A StreamHandler is always added for console output. When ``enable_logs`` is True, a FileHandler is also added, writing to a timestamped file in the configured log directory. The ``force=True`` argument to basicConfig ensures that existing handlers are replaced, allowing correct re-configuration at runtime.

        Returns:
            None
        """
        # Determine the log level based on the verbose flag in the config. 
        level = logging.DEBUG if self.config.verbose else logging.INFO

        # Define a consistent log message format that includes the timestamp, log level, logger name, and message. 
        fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

        # Extract the date format for logging timestamps
        datefmt = "%Y-%m-%d %H:%M:%S"

        # Initialize the list of logging handlers with a StreamHandler for console output. 
        handlers: list[logging.Handler] = [logging.StreamHandler()]

        if self.config.enable_logs:
            # Specify the log directory, resolving it relative to the config if necessary.
            log_dir = self.config.resolve_relative_path(self.config.log_dir)

            # Ensure the log directory exists, creating it if necessary. 
            os.makedirs(log_dir, exist_ok=True)

            # Extract the verification domain string
            vx_str = "_".join(self.config.vxdomain)

            # Generate a timestamp string for the log file name in the format YYYYmmdd_HHMMSS
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Construct the log file path using the experiment name, verification domain, and timestamp
            log_path = os.path.join(
                log_dir,
                f"{self.config.experiment_name}_{vx_str}_{ts}.log",
            )

            # Append a FileHandler to the handlers list for logging to a file when enabled in the config. 
            handlers.append(logging.FileHandler(log_path))

        # Configure the root logger with the determined level, format, date format, and handlers. 
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)


    def build_work_units(self) -> List[Dict[str, Any]]:
        """
        Build and return the complete list of coarse-grained (cycle × region) work-units. Each work-unit is a dictionary with keys ``cycle_start``, ``region_name``, and ``mask_path``, representing one combination of forecast cycle and verification domain. The full (threshold × window) parameter sweep is handled inside execute_work_unit, so work-unit granularity is intentionally coarse to maximise data reuse. Only domains listed in config.vxdomain are included in the output.

        Returns:
            List[Dict[str, Any]]: List of work-unit dictionaries, one per (cycle, region) combination.
        """
        # Extract the config for easy access within this method.
        config = self.config

        # Generate the list of cycle start times 
        cycles = list(
            iterate_forecast_cycle_starts(
                config.initial_cycle_start,
                config.final_cycle_start,
                config.cycle_interval,
            )
        )

        # Identify the subset of regions from the config that are included in the verification domain list
        regions = {
            name: path
            for name, path in config.regions.items()
            if name in config.vxdomain
        }
        
        # Initialize an empty list to hold all work-units
        work_units: List[Dict[str, Any]] = []

        # Construct the full list of work-units by iterating over all cycle starts and regions
        for cycle_start in cycles:
            for region_name, mask_file_path in regions.items():
                work_units.append(
                    {
                        "cycle_start": cycle_start,
                        "region_name": region_name,
                        "mask_path": mask_file_path,
                    }
                )
        
        # Return the complete list of work-units to be processed
        return work_units


    def _compute_metrics_for_valid_time(
        self,
        valid_time: datetime.datetime,
        cycle_init_str: str,
        cycle_start: datetime.datetime,
        region_mask: Any,
    ) -> Dict[Tuple[float, int], Dict[str, float]]:
        """
        This helper drives the per-timestep pipeline for a single valid_time: it loads accumulated forecast and observation fields via :class:`FileManager`, prepares them with :class:`DataValidator` (regridding and masking), optionally writes intermediate debug fields, and delegates to :meth:`PerfMetrics.compute_fss_batch` to calculate all threshold/window combinations. The returned dictionary maps ``(threshold, window)`` to the computed metrics for that valid time.

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window.
            cycle_init_str (str): Cycle initialisation string in ``YYYYmmddHH`` format.
            cycle_start (datetime.datetime): Cycle start time for naming and logging.
            region_mask (Any): Region mask DataArray used for domain masking.

        Returns:
            Dict[Tuple[float, int], Dict[str, float]]: Mapping of ``(threshold, window)`` to metric dicts.
        """
        # Extract the config for easy access within this method.
        config = self.config

        # Compute the accumulated forecast for the valid time and cycle using FileManager. 
        accumulated_forecast = self.file_manager.accumulate_forecasts(valid_time, cycle_init_str)

        # Compute the accumulated observation for the valid time using FileManager. 
        accumulated_observation = self.file_manager.accumulate_observations(valid_time)

        # Prepare the forecast and observation data by regridding to a common grid, applying the region mask, and performing any necessary unit conversions. 
        forecast_data, observation_data = self.data_validator.prepare(
            accumulated_forecast, accumulated_observation, region_mask, valid_time,
        )

        # Save intermediate fields for debugging if enabled in the config.
        if config.save_intermediate:
            self.file_manager.save_intermediate_precip(
                forecast_data, observation_data, cycle_start, valid_time,
            )

        # Return the full batch of FSS metrics for all (threshold, window) combinations for this valid time. 
        return self.perf_metrics.compute_fss_batch(
            forecast_data, observation_data,
            experiment_name=config.experiment_name,
            cycle_start=cycle_start,
            valid_time=valid_time,
            save_intermediate=config.save_intermediate,
        )

    def _persist_cycle_results(
        self,
        results_by_parameter: Dict[Tuple[float, int], List[Dict[str, float]]],
        cycle_start: datetime.datetime,
        region_name: str,
        cycle_init_str: str,
        num_valid_times: int,
    ) -> None:
        """
        This function writes one NetCDF result file per (threshold, window) combination using :class:`FileManager.save_fss_results`. It guards against the empty-result case by logging a warning rather than raising an exception, and logs a completion summary including the number of valid times and parameter combinations processed.

        Parameters:
            results_by_parameter (Dict[Tuple[float,int], List[Dict[str,float]]]): Collected metrics per (threshold, window) across the cycle.
            cycle_start (datetime.datetime): Cycle start time used for file paths.
            region_name (str): Verification domain name.
            cycle_init_str (str): Cycle initialisation string in ``YYYYmmddHH`` format.
            num_valid_times (int): Number of valid times attempted in the cycle.

        Returns:
            None
        """
        # Calculate the total number of parameter combinations for logging purposes.
        num_param_combinations = len(self.config.thresholds) * len(self.config.window_sizes)

        # Handle the case where all valid times failed and no results were collected
        if not results_by_parameter:
            logger.warning(
                "No FSS results for %s / %s — all valid times failed",
                cycle_init_str, region_name,
            )
            return

        # Iterate over the collected results for each (threshold, window) combination and save them in batch using FileManager. 
        for (threshold, window_size), metrics_list in results_by_parameter.items():
            self.file_manager.save_fss_results(
                metrics_list, cycle_start, region_name, threshold, window_size,
            )

        # Log a summary of the completed cycle and region for visibility into progress and performance.
        logger.info(
            "Completed %s / %s — %d valid times × %d param combos",
            cycle_init_str, region_name, num_valid_times, num_param_combinations,
        )

    def execute_work_unit(self, work_unit: Dict[str, Any]) -> None:
        """
        Execute a single (cycle, region) work-unit and persist all FSS results. For each valid time within the cycle, forecast and observation data are loaded and prepared exactly once, then the full (threshold × window) FSS parameter sweep is performed via PerfMetrics.compute_fss_batch. This design eliminates the redundant I/O and regridding that would occur if each (threshold, window) pair were a separate work-unit. FSS values are accumulated per parameter combination and saved in batch at the end of the cycle.

        Parameters:
            work_unit (dict): Work-unit dictionary with keys ``cycle_start``, ``region_name``, and ``mask_path``.

        Returns:
            None
        """
        from .utils import iterate_valid_times

        # Extract the config for easy access within this method.
        config = self.config

        # Extract the cycle start time from the work unit for logging, file naming, and data loading. 
        cycle_start: datetime.datetime = work_unit["cycle_start"]

        # Extract the region name from the work unit for logging and file naming.
        region_name: str = work_unit["region_name"]

        # Identify the mask file path for the current region from the work unit. 
        mask_file_path: str = work_unit["mask_path"]

        # Format the cycle initialisation string for logging and file naming. 
        cycle_init_str = cycle_start.strftime("%Y%m%d%H")

        # Calculate the total number of parameter combinations for logging purposes. 
        num_param_combinations = len(config.thresholds) * len(config.window_sizes)

        # Log the start of processing for this cycle and region, including the number of parameter combinations to be computed.
        logger.info(
            "Cycle %s | Region %s | %d thresh × %d win = %d combos",
            cycle_start.strftime("%Y-%m-%d %H:%M"), region_name,
            len(config.thresholds), len(config.window_sizes), num_param_combinations,
        )

        # Get the region mask once per cycle and reuse for all valid times to avoid redundant I/O and processing.
        region_mask, _ = self.file_manager.load_region_mask(
            self.config.resolve_mask_path(mask_file_path)
        )

        # Generate the list of valid times for this cycle based on the forecast length and step defined in the config. 
        valid_times = list(
            iterate_valid_times(cycle_start, cycle_start + config.forecast_length, config.forecast_step)
        )

        # Initialize a dictionary to accumulate FSS results by (threshold, window) across all valid times in the cycle. 
        results_by_parameter: Dict[Tuple[float, int], List[Dict[str, float]]] = defaultdict(list)

        for valid_time in valid_times:
            try:
                # Compute all FSS metrics for the current valid time across the full parameter sweep.
                batch_results = self._compute_metrics_for_valid_time(
                    valid_time, cycle_init_str, cycle_start, region_mask,
                )

                # Accumulate results by parameter combination for batch persistence at the end of the cycle. 
                for (threshold, window_size), metrics_dict in batch_results.items():
                    results_by_parameter[(threshold, window_size)].append(metrics_dict)
            except Exception:
                # Catch all exceptions to ensure that one failed valid time doesn't prevent the entire cycle from being processed. 
                logger.warning(
                    "Skipping valid_time %s for cycle %s: missing data",
                    valid_time, cycle_init_str, exc_info=True,
                )

        # Persist all results for the cycle after processing all valid times, even if some valid times failed. 
        self._persist_cycle_results(results_by_parameter, cycle_start, region_name, cycle_init_str, len(valid_times))


    def run(self) -> None:
        """
        Build all work-units and execute them sequentially in the current process. This method provides a simple serial execution path without requiring ParallelProcessor. It is suitable for small experiments or debugging sessions where parallel overhead is unnecessary. For production runs requiring MPI or multiprocessing parallelism, use ParallelProcessor.run with the unit list from build_work_units instead.

        Returns:
            None
        """
        # Log the start of the FSS computation with the experiment name for context
        logger.info("Starting FSS computation for experiment: %s", self.config.experiment_name)

        # Generate the complete list of work-units to be processed
        work_units = self.build_work_units()

        # Log the total number of work-units to be processed
        logger.info("Total work-units: %d", len(work_units))

        # Execute each work-unit sequentially, logging progress. 
        for work_unit in work_units:
            self.execute_work_unit(work_unit)

        # Log completion message after all work-units have been processed
        logger.info("FSS computation complete.")
