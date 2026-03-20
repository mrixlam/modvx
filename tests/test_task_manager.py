#!/usr/bin/env python3

"""
Unit tests for MODvx task manager.

This module contains unit tests for the `TaskManager` class, validating work-unit generation, logging setup, and the execution pipeline with mocked I/O. Tests exercise `build_work_units`, logging/file-handler creation, and `execute_work_unit` behaviour across multiple cycles and domains. Fixtures create minimal `ModvxConfig` instances rooted in temporary directories so the suite runs in isolation without external dependencies.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from modvx.config import ModvxConfig
from modvx.task_manager import TaskManager


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> ModvxConfig:
    """
    Construct a minimal ModvxConfig suitable for TaskManager unit tests. The configuration covers two forecast cycles 24 hours apart with a single GLOBAL domain, allowing build_work_units to produce a known two-unit list without requiring real data. All directories are rooted in the pytest temporary path so no real filesystem writes occur.

    Parameters:
        tmp_path (Path): Pytest-supplied per-test temporary directory.

    Returns:
        ModvxConfig: Configuration object with two cycles, one domain, and minimal required fields.
    """
    return ModvxConfig(
        base_dir=str(tmp_path),
        experiment_name="test_exp",
        initial_cycle_start=datetime.datetime(2024, 9, 17, 0, 0),
        final_cycle_start=datetime.datetime(2024, 9, 18, 0, 0),
        cycle_interval_hours=24,
        forecast_length_hours=6,
        forecast_step_hours=1,
        vxdomain=["GLOBAL"],
        thresholds=[90.0],
        window_sizes=[3],
    )


class TestTaskManagerLogging:
    """ Tests for TaskManager._setup_logging verifying verbose flag behavior and log file creation. These tests confirm that the logging configuration correctly sets the root logger level based on the verbose flag and that log files are created when enabled. By using temporary directories and minimal configurations, we can validate the logging setup without affecting the real filesystem. """

    def test_verbose_sets_debug(self, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that TaskManager sets the root logger to DEBUG level when verbose=True. The verbose flag is intended for developers and CI runs that need detailed trace output. This test constructs a TaskManager with the flag enabled and confirms the root logger level is lowered from the default INFO to DEBUG.

        Returns:
            None
        """
        tmp_cfg = ModvxConfig(**{**tmp_cfg.__dict__, "verbose": True})
        TaskManager(tmp_cfg)
        assert logging.getLogger().level == logging.DEBUG

    def test_file_handler_created(self, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that TaskManager creates a log file on disk when enable_logs=True. Log file persistence is useful for post-run debugging and monitoring of long pipeline runs. This test enables the flag and confirms at least one .log file was written to the configured log directory under the base directory.

        Returns:
            None
        """
        tmp_cfg = ModvxConfig(**{**tmp_cfg.__dict__, "enable_logs": True})
        TaskManager(tmp_cfg)
        log_dir = Path(tmp_cfg.base_dir) / tmp_cfg.log_dir
        log_files = list(log_dir.rglob("*.log"))
        assert len(log_files) >= 1


class TestBuildWorkUnits:
    """ Tests for TaskManager.build_work_units verifying correct unit count, required keys, and domain filtering. These tests confirm that the method produces the expected number of work units based on the configured cycles and domains, that each unit contains the necessary keys for execution, and that domain filtering is respected. By using minimal configurations and temporary directories, we can validate the work unit generation logic without relying on real data or filesystem state. """

    def test_correct_count(self, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that build_work_units returns the correct number of units for two cycles and one domain. With initial and final cycle 24 hours apart and a 24-hour interval, the cycle list has exactly two entries. Multiplied by one domain (GLOBAL), the expected work-unit count is 2.

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)
        units = tm.build_work_units()
        assert len(units) == 2

    def test_unit_keys(self, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that every work unit returned by build_work_units contains the required dictionary keys. Each unit must have cycle_start, region_name, and mask_path so that execute_work_unit can unpack them without a KeyError. This test iterates over all produced units to ensure the contract holds for every cycle-domain combination.

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)
        units = tm.build_work_units()
        for u in units:
            assert "cycle_start" in u
            assert "region_name" in u
            assert "mask_path" in u

    def test_multiple_domains(self, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that build_work_units produces the correct count when multiple verification domains are configured. With two domains (GLOBAL and TROPICS) and two cycles, the expected unit count is 4. This test confirms the domain loop inside build_work_units iterates correctly over all entries in vxdomain without skipping or duplicating any.

        Returns:
            None
        """
        tmp_cfg = ModvxConfig(
            **{**tmp_cfg.__dict__, "vxdomain": ["GLOBAL", "TROPICS"]}
        )
        tm = TaskManager(tmp_cfg)
        units = tm.build_work_units()
        # 2 cycles × 2 domains = 4
        assert len(units) == 4

    def test_filters_to_vxdomain(self, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that build_work_units only produces units for domains listed in the vxdomain configuration. When vxdomain=['GLOBAL'], all returned units must have region_name == 'GLOBAL'. This guards against accidental inclusion of all available mask files when the user has restricted the domain set for a focused verification run.

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)
        units = tm.build_work_units()
        assert all(u["region_name"] == "GLOBAL" for u in units)


class TestExecuteWorkUnit:
    """ Tests for TaskManager.execute_work_unit verifying successful execution, error handling, and I/O interactions. These tests use mocked dependencies to isolate the execution logic, ensuring that work units are processed correctly, errors are handled gracefully, and results are saved as expected. """

    def test_successful_execution(self, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that execute_work_unit completes without error and calls save_fss_results when all I/O is mocked. This integration-style unit test mocks every external dependency — mask loading, forecast and observation accumulation, and data preparation — so the execution logic can be exercised without real files. The save_fss_results mock call count must be positive to confirm the metrics pipeline ran.

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)

        # Mock all I/O
        mock_mask = xr.DataArray(
            np.ones((5, 5)),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )
        fcst = xr.DataArray(
            np.random.rand(5, 5),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )
        obs = fcst.copy()

        with patch.object(tm.file_manager, "load_region_mask", return_value=(mock_mask, "mask")), \
             patch.object(tm.file_manager, "accumulate_forecasts_precip_accum", return_value=fcst), \
             patch.object(tm.file_manager, "accumulate_observations_precip_accum", return_value=obs), \
             patch.object(tm.data_validator, "prepare", return_value=(fcst, obs)), \
             patch.object(tm.file_manager, "save_fss_results") as mock_save_fss, \
             patch.object(tm.file_manager, "save_contingency_results") as mock_save_cont:

            unit = {
                "cycle_start": datetime.datetime(2024, 9, 17, 0, 0),
                "region_name": "GLOBAL",
                "mask_path": "G004_GLOBAL.nc",
            }
            tm.execute_work_unit(unit)
            assert mock_save_fss.call_count > 0
            assert mock_save_cont.call_count > 0

    def test_skips_failing_valid_times(self, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that execute_work_unit silently skips a valid time that raises FileNotFoundError and continues. When a forecast file is missing for a specific valid time, processing should log the error and proceed to the next valid time rather than aborting the entire work unit. This test confirms that save_fss_results is never called when all valid times fail.

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)

        mock_mask = xr.DataArray(
            np.ones((5, 5)),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )

        with patch.object(tm.file_manager, "load_region_mask", return_value=(mock_mask, "mask")), \
             patch.object(tm.file_manager, "accumulate_forecasts_precip_accum", side_effect=FileNotFoundError), \
             patch.object(tm.file_manager, "save_fss_results") as mock_save_fss, \
             patch.object(tm.file_manager, "save_contingency_results") as mock_save_cont:

            unit = {
                "cycle_start": datetime.datetime(2024, 9, 17, 0, 0),
                "region_name": "GLOBAL",
                "mask_path": "G004_GLOBAL.nc",
            }
            tm.execute_work_unit(unit)
            # All valid times failed, so no save occurs
            mock_save_fss.assert_not_called()
            mock_save_cont.assert_not_called()

    def test_save_intermediate_flag(self, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that execute_work_unit calls save_intermediate_precip when save_intermediate=True is configured. The intermediate save flag is used to persist forecast and observation arrays before FSS computation for debugging purposes. This test enables the flag, mocks all I/O, and asserts the intermediate save method is called at least once confirming proper propagation from config to the pipeline.

        Returns:
            None
        """
        tmp_cfg = ModvxConfig(**{**tmp_cfg.__dict__, "save_intermediate": True})
        tm = TaskManager(tmp_cfg)

        mock_mask = xr.DataArray(
            np.ones((5, 5)),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )
        fcst = xr.DataArray(
            np.random.rand(5, 5),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )
        obs = fcst.copy()

        with patch.object(tm.file_manager, "load_region_mask", return_value=(mock_mask, "mask")), \
             patch.object(tm.file_manager, "accumulate_forecasts_precip_accum", return_value=fcst), \
             patch.object(tm.file_manager, "accumulate_observations_precip_accum", return_value=obs), \
             patch.object(tm.data_validator, "prepare", return_value=(fcst, obs)), \
             patch.object(tm.file_manager, "save_intermediate_precip") as mock_precip, \
             patch.object(tm.file_manager, "save_fss_results"), \
             patch.object(tm.file_manager, "save_contingency_results"):

            unit = {
                "cycle_start": datetime.datetime(2024, 9, 17, 0, 0),
                "region_name": "GLOBAL",
                "mask_path": "G004_GLOBAL.nc",
            }
            tm.execute_work_unit(unit)
            assert mock_precip.call_count > 0


class TestTaskManagerRun:
    """ Tests for TaskManager.run serial execution path verifying that each work unit is executed exactly once. These tests confirm that the run method correctly iterates over all work units produced by build_work_units and calls execute_work_unit for each one, ensuring complete and accurate processing of the pipeline. """

    def test_run_calls_execute(self, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that TaskManager.run() calls execute_work_unit once for every work unit built by build_work_units. The run method is the public entry point for serial pipeline execution and must not skip or batch any units. For two cycles and one domain this means exactly two execute_work_unit calls.

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)
        with patch.object(tm, "execute_work_unit") as mock_exec:
            tm.run()
            assert mock_exec.call_count == 2  # 2 cycles × 1 domain


class TestPrecipAccumValidation:
    """ Tests for precip_accum_hours validation in TaskManager.__init__ verifying correct handling of multiples and default values. These tests confirm that the constructor raises appropriate errors when precip_accum_hours is not a multiple of forecast_step_hours or observation_interval_hours, and that the default value passes validation. """

    def test_valid_multiple_passes(self, tmp_path: Path) -> None:
        """precip_accum_hours=3 with forecast_step_hours=1 and obs_interval=1 should pass."""
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            forecast_step_hours=1,
            observation_interval_hours=1,
            precip_accum_hours=3,
            vxdomain=["GLOBAL"],
            thresholds=[90.0],
            window_sizes=[3],
        )
        TaskManager(cfg) 

    def test_invalid_step_multiple_raises(self, tmp_path: Path) -> None:
        """precip_accum_hours=5 with forecast_step_hours=3 should raise ValueError."""
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            forecast_step_hours=3,
            observation_interval_hours=1,
            precip_accum_hours=5,
            vxdomain=["GLOBAL"],
            thresholds=[90.0],
            window_sizes=[3],
        )
        with pytest.raises(ValueError, match="multiple of forecast_step_hours"):
            TaskManager(cfg)

    def test_invalid_obs_multiple_raises(self, tmp_path: Path) -> None:
        """precip_accum_hours=5 with observation_interval_hours=2 should raise ValueError."""
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            forecast_step_hours=1,
            observation_interval_hours=2,
            precip_accum_hours=5,
            vxdomain=["GLOBAL"],
            thresholds=[90.0],
            window_sizes=[3],
        )
        with pytest.raises(ValueError, match="multiple of observation_interval_hours"):
            TaskManager(cfg)

    def test_zero_default_passes(self, tmp_cfg: ModvxConfig) -> None:
        """precip_accum_hours=0 (default) should always pass validation."""
        TaskManager(tmp_cfg)  


class TestPrecipAccumStride:
    """ Tests verifying that valid-time iteration uses precip_accum as stride. These tests confirm that the TaskManager correctly calculates the number of valid times based on the forecast length and precip_accum_hours, ensuring that the stride is applied consistently across all cycles and domains. """

    def test_stride_uses_precip_accum(self, tmp_path: Path) -> None:
        """With forecast_length=6h, step=1h, precip_accum=3h, expect 2 valid times."""
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            experiment_name="test_exp",
            initial_cycle_start=datetime.datetime(2024, 9, 17, 0, 0),
            final_cycle_start=datetime.datetime(2024, 9, 17, 0, 0),
            cycle_interval_hours=24,
            forecast_length_hours=6,
            forecast_step_hours=1,
            precip_accum_hours=3,
            vxdomain=["GLOBAL"],
            thresholds=[90.0],
            window_sizes=[3],
        )
        tm = TaskManager(cfg)

        mock_mask = xr.DataArray(
            np.ones((5, 5)),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )
        fcst = xr.DataArray(
            np.random.rand(5, 5),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )
        obs = fcst.copy()

        with patch.object(tm.file_manager, "load_region_mask", return_value=(mock_mask, "mask")), \
             patch.object(tm.file_manager, "accumulate_forecasts_precip_accum", return_value=fcst) as mock_fcst, \
             patch.object(tm.file_manager, "accumulate_observations_precip_accum", return_value=obs) as mock_obs, \
             patch.object(tm.data_validator, "prepare", return_value=(fcst, obs)), \
             patch.object(tm.file_manager, "save_fss_results"), \
             patch.object(tm.file_manager, "save_contingency_results"):

            unit = {
                "cycle_start": datetime.datetime(2024, 9, 17, 0, 0),
                "region_name": "GLOBAL",
                "mask_path": "G004_GLOBAL.nc",
            }
            tm.execute_work_unit(unit)
            # 6h / 3h stride = 2 valid times
            assert mock_fcst.call_count == 2
            assert mock_obs.call_count == 2
