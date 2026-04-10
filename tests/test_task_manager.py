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

import numpy as np
import pytest
import xarray as xr

from modvx.config import ModvxConfig
from modvx.task_manager import TaskManager


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> ModvxConfig:
    """
    This fixture provides a minimal ModvxConfig for testing TaskManager methods. It configures two cycles 24 hours apart, one verification domain (GLOBAL), and the required fields for work-unit generation. By using the tmp_path fixture, it ensures that all file operations during tests are confined to a temporary directory that is cleaned up after the test run. 

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
    """ Tests for TaskManager._setup_logging verifying verbose flag behavior and log file creation. """

    def test_verbose_sets_debug(self: "TestTaskManagerLogging", 
                                tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that when the verbose flag is set to True in the configuration, TaskManager._setup_logging configures the root logger to DEBUG level. This ensures that detailed debug information will be printed to the console during execution, which is essential for troubleshooting and understanding the internal workings of the pipeline.

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with the verbose flag set to True.

        Returns:
            None
        """
        tmp_cfg = ModvxConfig(**{**tmp_cfg.__dict__, "verbose": True})
        TaskManager(tmp_cfg)
        assert logging.getLogger().level == logging.DEBUG

    def test_file_handler_created(self: "TestTaskManagerLogging", 
                                  tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that when the enable_logs flag is set to True in the configuration, TaskManager._setup_logging creates a file handler that writes log messages to a file in the specified log directory. It checks that at least one .log file is created in the log directory, confirming that logging to file is properly set up. This ensures that log messages are persisted for later review, which is important for debugging and monitoring long-running tasks.

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with the enable_logs flag set to True.

        Returns:
            None
        """
        tmp_cfg = ModvxConfig(**{**tmp_cfg.__dict__, "enable_logs": True})
        TaskManager(tmp_cfg)
        log_dir = Path(tmp_cfg.base_dir) / tmp_cfg.log_dir
        log_files = list(log_dir.rglob("*.log"))
        assert len(log_files) >= 1


class TestBuildWorkUnits:
    """ Tests for TaskManager.build_work_units verifying correct unit count, required keys, and domain filtering. """

    def test_correct_count(self: "TestBuildWorkUnits", 
                           tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that TaskManager.build_work_units produces the correct number of work units based on the configuration. With two cycles (2024-09-17 and 2024-09-18) and one domain (GLOBAL), the expected number of work units is 2. This confirms that the method correctly iterates over the cycle range and applies the domain filter to generate a unit for each cycle-domain combination. 

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with two cycles and one domain.

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)
        units = tm.build_work_units()
        assert len(units) == 2

    def test_unit_keys(self: "TestBuildWorkUnits", 
                       tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that each work unit produced by TaskManager.build_work_units contains the required keys: 'cycle_start', 'region_name', and 'mask_path'. These keys are essential for the execution of the work unit, as they provide the necessary information about the cycle time, verification domain, and mask file path. The test iterates through all generated units and asserts that each key is present, ensuring that the work units are properly structured for downstream processing. 

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with two cycles and one domain.

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)
        units = tm.build_work_units()
        for u in units:
            assert "cycle_start" in u
            assert "region_name" in u
            assert "mask_path" in u

    def test_multiple_domains(self: "TestBuildWorkUnits", 
                              tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that TaskManager.build_work_units correctly generates work units for multiple verification domains when specified in the configuration. By setting vxdomain to include both "GLOBAL" and "TROPICS", the expected number of work units should be 4 (2 cycles × 2 domains). This confirms that the method properly iterates over all specified domains for each cycle, ensuring comprehensive coverage of the configured verification regions. 

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with two cycles and two domains.

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

    def test_filters_to_vxdomain(self: "TestBuildWorkUnits", 
                                 tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that TaskManager.build_work_units correctly filters work units to only include those matching the specified verification domains in the configuration. By setting vxdomain to ["GLOBAL"], the test asserts that all generated work units have "region_name" equal to "GLOBAL". This ensures that the method respects the domain filter and does not produce units for any unspecified regions, which is crucial for targeted evaluation and resource management. 

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with two cycles and one domain (GLOBAL).

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)
        units = tm.build_work_units()
        assert all(u["region_name"] == "GLOBAL" for u in units)


class TestExecuteWorkUnit:
    """ Tests for TaskManager.execute_work_unit verifying successful execution, error handling, and I/O interactions. """

    def test_successful_execution(self: "TestExecuteWorkUnit", 
                                  tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that TaskManager.execute_work_unit successfully processes a work unit when all I/O operations are mocked to return valid data. It mocks the loading of the region mask, accumulation of forecasts and observations, preparation of data for validation, and saving of FSS and contingency results. The test asserts that the save methods are called at least once, confirming that the execution pipeline completes without errors and reaches the result-saving stage. This ensures that the method can handle a typical successful execution scenario end-to-end. 

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with two cycles and one domain, used to initialize the TaskManager for testing.

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

        fss_calls: list = []
        cont_calls: list = []

        tm.file_manager.load_region_mask = lambda *a, **kw: (mock_mask, "mask")
        tm.file_manager.accumulate_forecasts_precip_accum = lambda *a, **kw: fcst
        tm.file_manager.accumulate_observations_precip_accum = lambda *a, **kw: obs
        tm.data_validator.prepare = lambda *a, **kw: (fcst, obs)
        tm.file_manager.save_fss_results = lambda *a, **kw: fss_calls.append(True)
        tm.file_manager.save_contingency_results = lambda *a, **kw: cont_calls.append(True)

        unit = {
            "cycle_start": datetime.datetime(2024, 9, 17, 0, 0),
            "region_name": "GLOBAL",
            "mask_path": "G004_GLOBAL.nc",
        }
        tm.execute_work_unit(unit)
        assert len(fss_calls) > 0
        assert len(cont_calls) > 0

    def test_skips_failing_valid_times(self: "TestExecuteWorkUnit", 
                                       tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that TaskManager.execute_work_unit correctly handles scenarios where all valid times fail due to missing forecast data. By mocking the accumulate_forecasts_precip_accum method to raise a FileNotFoundError, it simulates a situation where no valid forecast data is available for any valid time. The test asserts that the save_fss_results and save_contingency_results methods are not called, confirming that the execution pipeline properly skips saving results when all valid times fail, which is essential for robust error handling and resource management. 

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with two cycles and one domain, used to initialize the TaskManager for testing.

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)

        mock_mask = xr.DataArray(
            np.ones((5, 5)),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )

        def _raise_fnf(*args, **kwargs):
            raise FileNotFoundError()

        fss_calls: list = []
        cont_calls: list = []

        tm.file_manager.load_region_mask = lambda *a, **kw: (mock_mask, "mask")
        tm.file_manager.accumulate_forecasts_precip_accum = _raise_fnf
        tm.file_manager.save_fss_results = lambda *a, **kw: fss_calls.append(True)
        tm.file_manager.save_contingency_results = lambda *a, **kw: cont_calls.append(True)

        unit = {
            "cycle_start": datetime.datetime(2024, 9, 17, 0, 0),
            "region_name": "GLOBAL",
            "mask_path": "G004_GLOBAL.nc",
        }
        tm.execute_work_unit(unit)
        # All valid times failed, so no save occurs
        assert len(fss_calls) == 0
        assert len(cont_calls) == 0

    def test_save_intermediate_flag(self: "TestExecuteWorkUnit", 
                                    tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that when the save_intermediate flag is set to True in the configuration, TaskManager.execute_work_unit calls the method to save intermediate precip accumulation results at least once during execution. By mocking the relevant I/O methods and asserting that the save_intermediate_precip method is called, it confirms that the execution pipeline respects the configuration setting to save intermediate results, which can be crucial for debugging and analysis of the accumulation process. 

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with two cycles and one domain, used to initialize the TaskManager for testing.

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

        precip_calls: list = []

        tm.file_manager.load_region_mask = lambda *a, **kw: (mock_mask, "mask")
        tm.file_manager.accumulate_forecasts_precip_accum = lambda *a, **kw: fcst
        tm.file_manager.accumulate_observations_precip_accum = lambda *a, **kw: obs
        tm.data_validator.prepare = lambda *a, **kw: (fcst, obs)
        tm.file_manager.save_intermediate_precip = lambda *a, **kw: precip_calls.append(True)
        tm.file_manager.save_fss_results = lambda *a, **kw: None
        tm.file_manager.save_contingency_results = lambda *a, **kw: None

        unit = {
            "cycle_start": datetime.datetime(2024, 9, 17, 0, 0),
            "region_name": "GLOBAL",
            "mask_path": "G004_GLOBAL.nc",
        }
        tm.execute_work_unit(unit)
        assert len(precip_calls) > 0


class TestTaskManagerRun:
    """ Tests for TaskManager.run serial execution path verifying that each work unit is executed exactly once. """

    def test_run_calls_execute(self: "TestTaskManagerRun", 
                               tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that TaskManager.run correctly iterates over all generated work units and calls the execute_work_unit method for each unit. By mocking the execute_work_unit method and asserting that it is called the expected number of times (2 cycles × 1 domain = 2), it confirms that the run method properly processes each work unit in the queue without skipping or duplicating any, ensuring that the entire set of configured tasks is executed as intended.

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with two cycles and one domain, used to initialize the TaskManager for testing.

        Returns:
            None
        """
        tm = TaskManager(tmp_cfg)

        executed_units: list = []
        tm.execute_work_unit = lambda unit: executed_units.append(unit)

        tm.run()
        assert len(executed_units) == 2  # 2 cycles × 1 domain


class TestPrecipAccumValidation:
    """ Tests for precip_accum_hours validation in TaskManager.__init__ verifying correct handling of multiples and default values. """

    def test_valid_multiple_passes(self: "TestPrecipAccumValidation", 
                                   tmp_path: Path) -> None:
        """
        This test verifies that when precip_accum_hours is set to a valid multiple of both forecast_step_hours and observation_interval_hours, TaskManager initializes without raising an error. By configuring forecast_step_hours=1, observation_interval_hours=1, and precip_accum_hours=3, it confirms that the validation logic correctly identifies 3 as a valid accumulation period, allowing the TaskManager to be instantiated successfully. This ensures that users can configure precip_accum_hours with valid multiples without encountering unnecessary errors. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for the base_dir in the configuration.

        Returns:
            None
        """
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

    def test_invalid_step_multiple_raises(self: "TestPrecipAccumValidation", 
                                          tmp_path: Path) -> None:
        """
        This test verifies that when precip_accum_hours is set to a value that is not a valid multiple of forecast_step_hours, TaskManager raises a ValueError with an appropriate message. By configuring forecast_step_hours=3 and precip_accum_hours=5, it confirms that the validation logic correctly identifies 5 as an invalid accumulation period relative to the step size, preventing the TaskManager from being instantiated and providing clear feedback on the configuration error. This ensures that users are guided to set precip_accum_hours to valid values that align with their forecast step configuration. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for the base_dir in the configuration.

        Returns:
            None
        """
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

    def test_invalid_obs_multiple_raises(self: "TestPrecipAccumValidation", 
                                         tmp_path: Path) -> None:
        """
        This test verifies that when precip_accum_hours is set to a value that is not a valid multiple of observation_interval_hours, TaskManager raises a ValueError with an appropriate message. By configuring observation_interval_hours=2 and precip_accum_hours=5, it confirms that the validation logic correctly identifies 5 as an invalid accumulation period relative to the observation interval, preventing the TaskManager from being instantiated and providing clear feedback on the configuration error. This ensures that users are guided to set precip_accum_hours to valid values that align with their observation interval configuration. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for the base_dir in the configuration.

        Returns:
            None
        """
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

    def test_zero_default_passes(self: "TestPrecipAccumValidation", 
                                 tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that when precip_accum_hours is set to 0 (default), TaskManager passes validation. By using the default value, it confirms that the validation logic correctly allows 0 as a valid accumulation period, ensuring that users can rely on the default configuration without encountering errors.

        Parameters:
            tmp_cfg (ModvxConfig): A configuration object with default settings.

        Returns:
            None
        """
        TaskManager(tmp_cfg)  


class TestPrecipAccumStride:
    """ Tests verifying that valid-time iteration uses precip_accum as stride. """

    def test_stride_uses_precip_accum(self: "TestPrecipAccumStride", 
                                      tmp_path: Path) -> None:
        """
        This test verifies that TaskManager.execute_work_unit uses precip_accum_hours as the stride for iterating over valid times when accumulating forecasts and observations. By configuring precip_accum_hours=3 and forecast_length_hours=6, it confirms that the method processes valid times at 0h and 3h (2 valid times) rather than every hour, ensuring that the accumulation logic correctly applies the specified accumulation period as the iteration step. This is crucial for accurate accumulation of precipitation over the defined intervals and for validating that the execution pipeline respects the configuration settings for accumulation. 
        
        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for the base_dir in the configuration.

        Returns:
            None
        """
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

        fcst_calls: list = []
        obs_calls: list = []

        tm.file_manager.load_region_mask = lambda *a, **kw: (mock_mask, "mask")
        tm.file_manager.accumulate_forecasts_precip_accum = lambda *a, **kw: (fcst_calls.append(a), fcst)[1]
        tm.file_manager.accumulate_observations_precip_accum = lambda *a, **kw: (obs_calls.append(a), obs)[1]
        tm.data_validator.prepare = lambda *a, **kw: (fcst, obs)
        tm.file_manager.save_fss_results = lambda *a, **kw: None
        tm.file_manager.save_contingency_results = lambda *a, **kw: None

        unit = {
            "cycle_start": datetime.datetime(2024, 9, 17, 0, 0),
            "region_name": "GLOBAL",
            "mask_path": "G004_GLOBAL.nc",
        }
        tm.execute_work_unit(unit)
        # 6h / 3h stride = 2 valid times
        assert len(fcst_calls) == 2
        assert len(obs_calls) == 2
