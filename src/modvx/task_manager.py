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
        self.config = config
        self.file_manager = FileManager(config)
        self.data_validator = DataValidator(config)
        self.perf_metrics = PerfMetrics(config)
        self._setup_logging()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """
        Configure Python's root logging system based on the current ModvxConfig settings. Log level is set to DEBUG when verbose mode is enabled, and INFO otherwise. A StreamHandler is always added for console output. When ``enable_logs`` is True, a FileHandler is also added, writing to a timestamped file in the configured log directory. The ``force=True`` argument to basicConfig ensures that existing handlers are replaced, allowing correct re-configuration at runtime.

        Returns:
            None
        """
        level = logging.DEBUG if self.config.verbose else logging.INFO
        fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        handlers: list[logging.Handler] = [logging.StreamHandler()]

        if self.config.enable_logs:
            log_dir = self.config.resolve_path(self.config.log_dir)
            os.makedirs(log_dir, exist_ok=True)
            vx_str = "_".join(self.config.vxdomain)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(
                log_dir,
                f"{self.config.experiment_name}_{vx_str}_{ts}.log",
            )
            handlers.append(logging.FileHandler(log_path))

        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)

    # ------------------------------------------------------------------
    # Work-unit generation
    # ------------------------------------------------------------------

    def build_work_units(self) -> List[Dict[str, Any]]:
        """
        Build and return the complete list of coarse-grained (cycle × region) work-units. Each work-unit is a dictionary with keys ``cycle_start``, ``region_name``, and ``mask_path``, representing one combination of forecast cycle and verification domain. The full (threshold × window) parameter sweep is handled inside execute_work_unit, so work-unit granularity is intentionally coarse to maximise data reuse. Only domains listed in config.vxdomain are included in the output.

        Returns:
            List[Dict[str, Any]]: List of work-unit dictionaries, one per (cycle, region)
                combination.
        """
        config = self.config
        cycles = list(
            iterate_forecast_cycle_starts(
                config.initial_cycle_start,
                config.final_cycle_start,
                config.cycle_interval,
            )
        )
        regions = {
            name: path
            for name, path in config.regions.items()
            if name in config.vxdomain
        }

        work_units: List[Dict[str, Any]] = []
        for cycle_start in cycles:
            for region_name, mask_file_path in regions.items():
                work_units.append(
                    {
                        "cycle_start": cycle_start,
                        "region_name": region_name,
                        "mask_path": mask_file_path,
                    }
                )
        return work_units

    # ------------------------------------------------------------------
    # Single work-unit execution
    # ------------------------------------------------------------------

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
        config = self.config

        accumulated_forecast = self.file_manager.accumulate_forecasts(valid_time, cycle_init_str)
        accumulated_observation = self.file_manager.accumulate_observations(valid_time)

        forecast_data, observation_data = self.data_validator.prepare(
            accumulated_forecast, accumulated_observation, region_mask, valid_time,
        )

        if config.save_intermediate:
            self.file_manager.save_intermediate_precip(
                forecast_data, observation_data, cycle_start, valid_time,
            )

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
            results_by_parameter (Dict[Tuple[float,int], List[Dict[str,float]]]): Collected
                metrics per (threshold, window) across the cycle.
            cycle_start (datetime.datetime): Cycle start time used for file paths.
            region_name (str): Verification domain name.
            cycle_init_str (str): Cycle initialisation string in ``YYYYmmddHH`` format.
            num_valid_times (int): Number of valid times attempted in the cycle.

        Returns:
            None
        """
        num_param_combinations = len(self.config.thresholds) * len(self.config.window_sizes)
        if not results_by_parameter:
            logger.warning(
                "No FSS results for %s / %s — all valid times failed",
                cycle_init_str, region_name,
            )
            return
        for (threshold, window_size), metrics_list in results_by_parameter.items():
            self.file_manager.save_fss_results(
                metrics_list, cycle_start, region_name, threshold, window_size,
            )
        logger.info(
            "Completed %s / %s — %d valid times × %d param combos",
            cycle_init_str, region_name, num_valid_times, num_param_combinations,
        )

    def execute_work_unit(self, work_unit: Dict[str, Any]) -> None:
        """
        Execute a single (cycle, region) work-unit and persist all FSS results. For each valid time within the cycle, forecast and observation data are loaded and prepared exactly once, then the full (threshold × window) FSS parameter sweep is performed via PerfMetrics.compute_fss_batch. This design eliminates the redundant I/O and regridding that would occur if each (threshold, window) pair were a separate work-unit. FSS values are accumulated per parameter combination and saved in batch at the end of the cycle.

        Parameters:
            work_unit (dict): Work-unit dictionary with keys ``cycle_start``, ``region_name``,
                and ``mask_path``.

        Returns:
            None
        """
        from .utils import iterate_valid_times

        config = self.config
        cycle_start: datetime.datetime = work_unit["cycle_start"]
        region_name: str = work_unit["region_name"]
        mask_file_path: str = work_unit["mask_path"]

        cycle_init_str = cycle_start.strftime("%Y%m%d%H")
        num_param_combinations = len(config.thresholds) * len(config.window_sizes)

        logger.info(
            "Cycle %s | Region %s | %d thresh × %d win = %d combos",
            cycle_start.strftime("%Y-%m-%d %H:%M"), region_name,
            len(config.thresholds), len(config.window_sizes), num_param_combinations,
        )

        region_mask, _ = self.file_manager.load_region_mask(
            self.config.resolve_mask_path(mask_file_path)
        )

        valid_times = list(
            iterate_valid_times(cycle_start, cycle_start + config.forecast_length, config.forecast_step)
        )
        results_by_parameter: Dict[Tuple[float, int], List[Dict[str, float]]] = defaultdict(list)

        for valid_time in valid_times:
            try:
                batch_results = self._compute_metrics_for_valid_time(
                    valid_time, cycle_init_str, cycle_start, region_mask,
                )
                for (threshold, window_size), metrics_dict in batch_results.items():
                    results_by_parameter[(threshold, window_size)].append(metrics_dict)
            except Exception:
                logger.warning(
                    "Skipping valid_time %s for cycle %s: missing data",
                    valid_time, cycle_init_str, exc_info=True,
                )

        self._persist_cycle_results(results_by_parameter, cycle_start, region_name, cycle_init_str, len(valid_times))

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Build all work-units and execute them sequentially in the current process. This method provides a simple serial execution path without requiring ParallelProcessor. It is suitable for small experiments or debugging sessions where parallel overhead is unnecessary. For production runs requiring MPI or multiprocessing parallelism, use ParallelProcessor.run with the unit list from build_work_units instead.

        Returns:
            None
        """
        logger.info("Starting FSS computation for experiment: %s", self.config.experiment_name)
        work_units = self.build_work_units()
        logger.info("Total work-units: %d", len(work_units))

        for work_unit in work_units:
            self.execute_work_unit(work_unit)

        logger.info("FSS computation complete.")
