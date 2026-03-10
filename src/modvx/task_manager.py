"""
Task orchestration for modvx.

``TaskManager`` is the top-level entry point that:

1. Reads a :class:`ModvxConfig` (YAML + optional CLI overrides).
2. Builds coarse-grained work-units (cycle × region).
3. Hands them to :class:`ParallelProcessor` for execution.
4. Manages per-run logging.

Each work-unit loads data **once** per valid-time and sweeps all
(threshold × window) combinations via
:meth:`PerfMetrics.compute_fss_batch`, eliminating redundant I/O.
"""

from __future__ import annotations

import datetime
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .config import ModvxConfig
from .data_validator import DataValidator
from .file_manager import FileManager
from .perf_metrics import PerfMetrics
from .utils import generate_forecast_cycles


logger = logging.getLogger(__name__)


class TaskManager:
    """
    Orchestrate the end-to-end FSS verification pipeline for a configured experiment.
    On construction, TaskManager instantiates FileManager, DataValidator, and PerfMetrics
    and configures Python logging according to the config flags. It exposes two key methods:
    build_work_units generates the (cycle × region) task list, and execute_work_unit processes
    a single task by loading data, preparing fields, computing FSS across all parameter
    combinations, and persisting results.

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
        Configure Python's root logging system based on the current ModvxConfig settings.
        Log level is set to DEBUG when verbose mode is enabled, and INFO otherwise. A
        StreamHandler is always added for console output. When ``enable_logs`` is True,
        a FileHandler is also added, writing to a timestamped file in the configured log
        directory. The ``force=True`` argument to basicConfig ensures that existing handlers
        are replaced, allowing correct re-configuration at runtime.

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
        Build and return the complete list of coarse-grained (cycle × region) work-units.
        Each work-unit is a dictionary with keys ``cycle_start``, ``region_name``, and
        ``mask_path``, representing one combination of forecast cycle and verification domain.
        The full (threshold × window) parameter sweep is handled inside execute_work_unit,
        so work-unit granularity is intentionally coarse to maximise data reuse. Only domains
        listed in config.vxdomain are included in the output.

        Returns:
            List[Dict[str, Any]]: List of work-unit dictionaries, one per (cycle, region)
                combination.
        """
        cfg = self.config
        cycles = list(
            generate_forecast_cycles(
                cfg.initial_cycle_start,
                cfg.final_cycle_start,
                cfg.cycle_interval,
            )
        )
        regions = {
            name: path
            for name, path in cfg.regions.items()
            if name in cfg.vxdomain
        }

        units: List[Dict[str, Any]] = []
        for cycle_start in cycles:
            for region_name, mask_path in regions.items():
                units.append(
                    {
                        "cycle_start": cycle_start,
                        "region_name": region_name,
                        "mask_path": mask_path,
                    }
                )
        return units

    # ------------------------------------------------------------------
    # Single work-unit execution
    # ------------------------------------------------------------------

    def execute_work_unit(self, unit: Dict[str, Any]) -> None:
        """
        Execute a single (cycle, region) work-unit and persist all FSS results.
        For each valid time within the cycle, forecast and observation data are loaded and
        prepared exactly once, then the full (threshold × window) FSS parameter sweep is
        performed via PerfMetrics.compute_fss_batch. This design eliminates the redundant
        I/O and regridding that would occur if each (threshold, window) pair were a separate
        work-unit. FSS values are accumulated per parameter combination and saved in batch
        at the end of the cycle.

        Parameters:
            unit (dict): Work-unit dictionary with keys ``cycle_start``, ``region_name``,
                and ``mask_path``.

        Returns:
            None
        """
        cfg = self.config
        cycle_start: datetime.datetime = unit["cycle_start"]
        region_name: str = unit["region_name"]
        mask_path: str = unit["mask_path"]

        init_string = cycle_start.strftime("%Y%m%d%H")
        forecast_end = cycle_start + cfg.forecast_length
        n_combos = len(cfg.thresholds) * len(cfg.window_sizes)

        logger.info(
            "Cycle %s | Region %s | %d thresh × %d win = %d combos",
            cycle_start.strftime("%Y-%m-%d %H:%M"),
            region_name,
            len(cfg.thresholds),
            len(cfg.window_sizes),
            n_combos,
        )

        # --- Load region mask (once per work-unit) -----------------------
        region_mask, _ = self.file_manager.load_region_mask(
            self.config.resolve_mask_path(mask_path)
        )

        # --- Accumulate FSS across valid-times ---------------------------
        from .utils import generate_valid_times

        valid_times = list(
            generate_valid_times(cycle_start, forecast_end, cfg.forecast_step)
        )
        # {(threshold, window_size): [metrics_dict_per_valid_time, …]}
        fss_results: Dict[Tuple[float, int], List[Dict[str, float]]] = defaultdict(list)

        for valid_time in valid_times:
            try:
                # Load data ONCE per valid-time
                forecast_accum = self.file_manager.accumulate_forecasts(
                    valid_time, init_string,
                )
                observation_accum = self.file_manager.accumulate_observations(
                    valid_time,
                )

                # Prepare (regrid + mask) ONCE per valid-time
                forecast_da, observation_da = self.data_validator.prepare(
                    forecast_accum, observation_accum, region_mask, valid_time,
                )

                # Optionally save intermediate precip fields
                if cfg.save_intermediate:
                    self.file_manager.save_intermediate_precip(
                        forecast_da, observation_da,
                        cycle_start, valid_time,
                    )

                # Sweep ALL (threshold, window) combos at once
                batch = self.perf_metrics.compute_fss_batch(
                    forecast_da, observation_da,
                    experiment_name=cfg.experiment_name,
                    cycle_start=cycle_start,
                    valid_time=valid_time,
                    save_intermediate=cfg.save_intermediate,
                )
                for (thresh, win), metrics in batch.items():
                    fss_results[(thresh, win)].append(metrics)
            except Exception:
                logger.warning(
                    "Skipping valid_time %s for cycle %s: missing data",
                    valid_time, init_string, exc_info=True,
                )

        # --- Save one file per (threshold, window) -----------------------
        if not fss_results:
            logger.warning(
                "No FSS results for %s / %s — all valid times failed",
                init_string, region_name,
            )
            return

        for (thresh, win), metrics_list in fss_results.items():
            self.file_manager.save_fss_results(
                metrics_list, cycle_start, region_name, thresh, win,
            )

        logger.info(
            "Completed %s / %s — %d valid times × %d param combos",
            init_string,
            region_name,
            len(valid_times),
            n_combos,
        )

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Build all work-units and execute them sequentially in the current process.
        This method provides a simple serial execution path without requiring ParallelProcessor.
        It is suitable for small experiments or debugging sessions where parallel overhead
        is unnecessary. For production runs requiring MPI or multiprocessing parallelism,
        use ParallelProcessor.run with the unit list from build_work_units instead.

        Returns:
            None
        """
        logger.info("Starting FSS computation for experiment: %s", self.config.experiment_name)
        units = self.build_work_units()
        logger.info("Total work-units: %d", len(units))

        for unit in units:
            self.execute_work_unit(unit)

        logger.info("FSS computation complete.")
