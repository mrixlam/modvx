#!/usr/bin/env python3

"""
Unit tests for MODvx file management utilities.

This module contains tests for the FileManager class and its associated helper functions, verifying correct file discovery, loading, caching, and saving behavior. The tests cover scenarios such as locating forecast and observation files based on configuration patterns, handling missing or malformed files gracefully, ensuring data is loaded into the expected xarray structures, and confirming that intermediate results are saved with the correct metadata and format. By isolating file management logic in these tests, we can ensure robust I/O operations that underpin the entire verification workflow.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""


import pytest
import datetime
import numpy as np
import xarray as xr

from modvx.config import ModvxConfig
from modvx.file_manager import FileManager


@pytest.fixture
def fm() -> FileManager:
    """
    This fixture provides a FileManager instance with a minimal hardcoded configuration suitable for testing. The base directory and MPAS grid file path are set to dummy values since the tests will focus on path construction logic rather than actual file I/O. By using a fixture, we can easily reuse this FileManager instance across multiple test cases without redundant setup code. 

    Parameters:
        None

    Returns:
        FileManager: FileManager backed by a minimal hardcoded configuration.
    """
    return FileManager(ModvxConfig(base_dir="/work", mpas_grid_file="grid/x1.10242.static.nc"))


class TestForecastPath:
    """ Tests for get_forecast_filepath verifying correct construction of MPAS diagnostic file paths based on valid time and forecast cycle. """

    def test_mpas_diag_path(self: "TestForecastPath", 
                            fm: FileManager) -> None:
        """
        This test verifies that the get_forecast_filepath method constructs the correct file path for an MPAS diagnostic file based on a given valid time and initialization time. By providing a specific valid time and initialization time, we can assert that the resulting path includes the expected filename format (e.g., "diag.YYYY-MM-DD_HH.MM.SS.nc") and directory structure (e.g., "ExtendedFC/YYYYMMDDHH"). This ensures that the method correctly translates temporal information into the standardized file naming convention used for MPAS diagnostics, which is critical for locating forecast output files during verification.

        Parameters:
            fm (FileManager): A FileManager instance with a known configuration for testing.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 17, 12, 0, 0)
        path = fm.get_forecast_filepath(vt, "2024091700")
        assert "diag.2024-09-17_12.00.00.nc" in path
        assert "ExtendedFC/2024091700" in path

    def test_mpas_diag_midnight(self: "TestForecastPath", 
                                fm: FileManager) -> None:
        """
        This test verifies that the get_forecast_filepath method correctly handles a valid time that falls at midnight (00:00 UTC). By providing a valid time of 2024-09-18 00:00:00 and an initialization time of 2024-09-17 00:00:00, we can assert that the resulting file path includes the expected filename format for the midnight valid time. This test ensures that the method correctly formats the timestamp in the filename even when the hour component is zero, which is important for consistent file naming and retrieval. 

        Parameters:
            fm (FileManager): A FileManager instance with a known configuration for testing.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 18, 0, 0, 0)
        path = fm.get_forecast_filepath(vt, "2024091700")
        assert "diag.2024-09-18_00.00.00.nc" in path


class TestObsHourIndex:
    """ Tests for get_observation_hour_index verifying correct mapping of valid times to hourly accumulation indices in FIMERG daily files. """

    def test_index_0(self: "TestObsHourIndex") -> None:
        """
        This test confirms that a valid time of 01:00 UTC maps to observation hour index 0 in the daily FIMERG file. Since FIMERG daily files use a 01:00–00:00 UTC accumulation window, the first hour of the day (01:00) corresponds to index 0, which represents the first band of the file. This test ensures that the method correctly identifies the starting point of the accumulation period and assigns the appropriate index for valid times that fall within the first hour after midnight. 

        Parameters:
            None

        Returns:
            None
        """
        # Time 01:00 → index 0
        assert FileManager.get_observation_hour_index(
            datetime.datetime(2025, 6, 14, 1, 0)
        ) == 0

    def test_midnight(self: "TestObsHourIndex") -> None:
        """
        This test verifies that a valid time of midnight (00:00 UTC) maps to observation hour index 23 in the daily FIMERG file. Since FIMERG daily files accumulate observations from 01:00 of the current day through 00:00 of the next day, the midnight hour belongs to the previous day's accumulation and is assigned index 23, which is the last band of the file. This test ensures that the method correctly handles the midnight boundary case and assigns the appropriate index for valid times that fall at the end of the accumulation period. 

        Parameters:
            None

        Returns:
            None
        """
        # Midnight → index 23 (belongs to previous day's file)
        assert FileManager.get_observation_hour_index(
            datetime.datetime(2025, 6, 15, 0, 0)
        ) == 23

    def test_hour_12(self: "TestObsHourIndex") -> None:
        """
        This test confirms that a valid time of 12:00 UTC maps to observation hour index 11 in the daily FIMERG file. Since the accumulation starts at 01:00 (index 0), the hour of 12:00 corresponds to the 12th hour of the accumulation period, which is index 11 (0-based indexing). This test ensures that the method correctly calculates the index for valid times that fall in the middle of the accumulation window, confirming that it accurately translates valid times into their corresponding indices within the daily file structure.

        Parameters:
            None

        Returns:
            None
        """
        assert FileManager.get_observation_hour_index(
            datetime.datetime(2025, 6, 14, 12, 0)
        ) == 11


class TestGroupObsTimes:
    """ Tests for group_observation_times_by_date verifying correct grouping of valid times into FIMERG daily file buckets, including midnight crossover handling. """

    def test_single_day(self: "TestGroupObsTimes") -> None:
        """
        This test confirms that valid times spanning a single calendar day are correctly grouped under the corresponding date key. By providing a range of valid times from 01:00 to 06:00 on June 14, 2025, we can assert that all these times are grouped under the "20250614" key and that the correct number of hours (6) is included in that group. This test ensures that the grouping logic correctly identifies valid times that belong to the same accumulation period and assigns them to the appropriate date-based group. 

        Parameters:
            None

        Returns:
            None
        """
        start = datetime.datetime(2025, 6, 14, 1)
        end = datetime.datetime(2025, 6, 14, 6)
        interval = datetime.timedelta(hours=1)
        groups = FileManager.group_observation_times_by_date(start, end, interval)
        assert "20250614" in groups
        assert len(groups["20250614"]) == 6  # hours 1-6

    def test_midnight_crossover(self: "TestGroupObsTimes") -> None:
        """
        This test verifies that valid times that span across midnight are correctly grouped into their respective date keys, with the midnight hour assigned to the previous day's group. By providing valid times from 23:00 on June 14, 2025, to 01:00 on June 15, 2025, we can assert that the times of 23:00 and 00:00 are grouped under the "20250614" key, while the time of 01:00 is grouped under the "20250615" key. This test ensures that the grouping logic correctly handles the transition at midnight and assigns valid times to the correct daily accumulation buckets based on FIMERG's convention. 

        Parameters:
            None

        Returns:
            None
        """
        start = datetime.datetime(2025, 6, 14, 23)
        end = datetime.datetime(2025, 6, 15, 1)
        interval = datetime.timedelta(hours=1)
        groups = FileManager.group_observation_times_by_date(start, end, interval)
        # 23:00 → 20250614, 00:00 → 20250614, 01:00 → 20250615
        assert "20250614" in groups
        assert "20250615" in groups
        assert len(groups["20250614"]) == 2  # 23:00 and midnight
        assert len(groups["20250615"]) == 1  # 01:00


class TestFcstCacheKey:
    """ Tests for _forecast_cache_key deterministic key generation. """

    def test_key_format(self: "TestFcstCacheKey", 
                        fm: FileManager) -> None:
        """
        This test confirms that the _forecast_cache_key method generates a cache key with the expected format based on the initialization time, valid time, and experiment name. By providing a specific initialization time of 2024-09-17 00:00:00 and a valid time of 2024-09-17 06:00:00, we can assert that the resulting key follows the pattern "fcst_accum_{experiment_name}_YYYYMMDDHH_YYYYMMDDHH_{accum_hours}h". This test ensures that the cache key generation logic correctly incorporates all relevant components and produces a consistent key format that can be used for caching forecast accumulations without ambiguity. 

        Parameters:
            fm (FileManager): A FileManager instance with a known configuration for testing.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 17, 6, 0, 0)
        key = fm._forecast_cache_key("2024091700", vt)
        exp = fm.config.experiment_name
        assert key == f"fcst_accum_{exp}_2024091700_2024091706_12h"

    def test_different_valid_times_yield_different_keys(self: "TestFcstCacheKey", 
                                                        fm: FileManager) -> None:
        """ 
        This test confirms that different valid times produce distinct cache keys, even when the initialization time is the same. By generating keys for two different valid times and asserting that they are not equal, we can ensure that the cache key generation logic correctly incorporates the valid time component. This is crucial for preventing cache collisions where different forecast outputs might overwrite each other due to identical keys. 

        Parameters:
            fm (FileManager): A FileManager instance with a known configuration for testing.

        Returns:
            None
        """
        vt1 = datetime.datetime(2024, 9, 17, 6)
        vt2 = datetime.datetime(2024, 9, 17, 7)
        assert fm._forecast_cache_key("2024091700", vt1) != fm._forecast_cache_key("2024091700", vt2)

    def test_different_init_times_yield_different_keys(self: "TestFcstCacheKey", 
                                                       fm: FileManager) -> None:
        """ 
        This test confirms that different initialization times produce distinct cache keys, even when the valid time is the same. By generating keys for two different initialization times and asserting that they are not equal, we can ensure that the cache key generation logic correctly incorporates the initialization time component. This is crucial for preventing cache collisions where different forecast outputs might overwrite each other due to identical keys. 

        Parameters:
            fm (FileManager): A FileManager instance with a known configuration for testing.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 17, 6)
        assert fm._forecast_cache_key("2024091700", vt) != fm._forecast_cache_key("2024091800", vt)

    def test_different_experiments_yield_different_keys(self: "TestFcstCacheKey") -> None:
        """
        This test confirms that different experiments produce distinct cache keys, even when the initialization time and valid time are the same. By generating keys for two different experiments and asserting that they are not equal, we can ensure that the cache key generation logic correctly incorporates the experiment name component. This is crucial for preventing cache collisions where different forecast outputs might overwrite each other due to identical keys.

        Parameters:
            None

        Returns:
            None
        """
        fm_a = FileManager(ModvxConfig(base_dir="/work", mpas_grid_file="grid/x1.10242.static.nc", experiment_name="exp_a"))
        fm_b = FileManager(ModvxConfig(base_dir="/work", mpas_grid_file="grid/x1.10242.static.nc", experiment_name="exp_b"))
        vt = datetime.datetime(2024, 9, 17, 6, 0, 0)
        assert fm_a._forecast_cache_key("2024091700", vt) != fm_b._forecast_cache_key("2024091700", vt)


class TestFcstMemCache:
    """ Tests for the in-memory forecast cache dict. """

    def test_cache_dict_exists(self: "TestFcstMemCache", 
                               fm: FileManager) -> None:
        """ 
        This test confirms that the in-memory forecast cache dictionary exists and is correctly initialized. By checking the presence and type of the cache dictionary, we can ensure that the FileManager instance is ready to store forecast data in memory for quick retrieval during verification.

        Parameters:
            fm (FileManager): A FileManager instance with a known configuration for testing.

        Returns:
            None
        """
        assert hasattr(fm, "_fcst_mem_cache")
        assert isinstance(fm._fcst_mem_cache, dict)
        assert len(fm._fcst_mem_cache) == 0


class TestAccumulateForecastsPrecipAccum:
    """ Tests for accumulate_forecasts_precip_accum multi-step accumulation. """

    def test_delegates_when_accum_equals_step(self: "TestAccumulateForecastsPrecipAccum") -> None:
        """
        This test verifies that when the configured precipitation accumulation hours are equal to the forecast step hours, the accumulate_forecasts_precip_accum method correctly delegates to the accumulate_forecasts method without performing additional summation. A call-tracking stub replaces the instance method directly; after the call the log is inspected to confirm the stub was invoked exactly once with the correct arguments and that the returned object is identical to the stub's return value.

        Parameters:
            None

        Returns:
            None
        """
        import numpy as np
        import xarray as xr

        cfg = ModvxConfig(base_dir="/work", mpas_grid_file="grid/x1.10242.static.nc",
                          forecast_step_hours=1, precip_accum_hours=0)
        fm_obj = FileManager(cfg)
        mock_da = xr.DataArray(np.ones((5, 5)), dims=["latitude", "longitude"])
        vt = datetime.datetime(2024, 9, 17, 0)

        call_log: list = []

        def fake_accumulate_forecasts(*args, 
                                      **kwargs) -> xr.DataArray:
            """
            This stub function simulates the behavior of accumulate_forecasts by logging its input arguments and returning a predefined xarray DataArray. By appending the received arguments to a call_log list, we can later verify that the method was called with the expected valid time and initialization string. The returned mock DataArray allows us to confirm that accumulate_forecasts_precip_accum correctly returns the result from accumulate_forecasts when the accumulation hours equal the step hours.

            Parameters:
                *args: Positional arguments passed to the stub.
                **kwargs: Keyword arguments passed to the stub.

            Returns:
                xr.DataArray: A predefined mock DataArray to simulate the output of accumulate_forecasts.
            """
            call_log.append(args)
            return mock_da

        fm_obj.accumulate_forecasts = fake_accumulate_forecasts
        result = fm_obj.accumulate_forecasts_precip_accum(vt, "2024091700")

        assert len(call_log) == 1
        assert call_log[0] == (vt, "2024091700")
        assert result is mock_da

    def test_sums_multiple_steps(self: "TestAccumulateForecastsPrecipAccum") -> None:
        """
        This test verifies that when the configured precipitation accumulation hours are greater than the forecast step hours, the accumulate_forecasts_precip_accum method correctly sums multiple forecast accumulations. A call-tracking stub is assigned directly on the instance; after the call the recorded argument list is compared against the expected valid-time sequence and the summed result is checked numerically.

        Parameters:
            None

        Returns:
            None
        """
        import numpy as np
        import xarray as xr

        cfg = ModvxConfig(base_dir="data/fcst", mpas_grid_file="grid/x1.10242.static.nc",
                          forecast_step_hours=1, precip_accum_hours=3)
        fm_obj = FileManager(cfg)
        mock_da = xr.DataArray(np.ones((5, 5)), dims=["latitude", "longitude"])
        vt = datetime.datetime(2024, 9, 17, 0)

        fm_obj._fcst_mem_cache.clear()
        fm_obj.config.cache_dir = None

        call_log: list = []

        def fake_accumulate_forecasts(*args, **kwargs) -> xr.DataArray:
            """ 
            This stub function simulates the behavior of accumulate_forecasts by logging its input arguments and returning a predefined xarray DataArray. By appending the received arguments to a call_log list, we can later verify that the method was called with the expected sequence of valid times and initialization string corresponding to the multi-step accumulation. The returned mock DataArray allows us to confirm that accumulate_forecasts_precip_accum correctly sums the results from multiple calls to accumulate_forecasts when the accumulation hours exceed the step hours. 

            Parameters:
                *args: Positional arguments passed to the stub.
                **kwargs: Keyword arguments passed to the stub.

            Returns:
                xr.DataArray: A predefined mock DataArray to simulate the output of accumulate_forecasts.
            """
            call_log.append(args)
            return mock_da

        fm_obj.accumulate_forecasts = fake_accumulate_forecasts
        result = fm_obj.accumulate_forecasts_precip_accum(vt, "2024091700")

        assert len(call_log) == 3
        assert call_log[0] == (datetime.datetime(2024, 9, 17, 0), "2024091700")
        assert call_log[1] == (datetime.datetime(2024, 9, 17, 1), "2024091700")
        assert call_log[2] == (datetime.datetime(2024, 9, 17, 2), "2024091700")
        assert float(result.values[0, 0]) == pytest.approx(3.0)


class TestAccumulateObsPrecipAccum:
    """ Tests for accumulate_observations_precip_accum multi-step accumulation. """

    def test_delegates_when_accum_equals_step(self: "TestAccumulateObsPrecipAccum") -> None:
        """
        This test verifies that when the configured precipitation accumulation hours are equal to the forecast step hours, the accumulate_observations_precip_accum method correctly delegates to the accumulate_observations method without performing additional summation. A call-tracking stub replaces the instance method directly; after the call the log is inspected to confirm the stub was invoked exactly once with the correct argument and that the returned object is identical to the stub's return value.

        Parameters:
            None

        Returns:
            None
        """
        import numpy as np
        import xarray as xr

        cfg = ModvxConfig(base_dir="data/fcst", mpas_grid_file="grid/x1.10242.static.nc",
                          forecast_step_hours=1, precip_accum_hours=0)
        fm_obj = FileManager(cfg)
        mock_da = xr.DataArray(np.ones((5, 5)), dims=["latitude", "longitude"])
        vt = datetime.datetime(2024, 9, 17, 0)

        call_log: list = []

        def fake_accumulate_observations(*args, 
                                         **kwargs) -> xr.DataArray:
            """
            This stub function simulates the behavior of accumulate_observations by logging its input arguments and returning a predefined xarray DataArray. By appending the received arguments to a call_log list, we can later verify that the method was called with the expected valid time. The returned mock DataArray allows us to confirm that accumulate_observations_precip_accum correctly delegates to accumulate_observations when the accumulation hours equal the step hours.

            Parameters:
                *args: Positional arguments passed to the stub.
                **kwargs: Keyword arguments passed to the stub.

            Returns:
                xr.DataArray: A predefined mock DataArray to simulate the output of accumulate_observations.
            """
            call_log.append(args)
            return mock_da

        fm_obj.accumulate_observations = fake_accumulate_observations
        result = fm_obj.accumulate_observations_precip_accum(vt)

        assert len(call_log) == 1
        assert call_log[0] == (vt,)
        assert result is mock_da


class TestFcstAccumPrecipCacheHit:
    """ Tests for in-memory cache hit in accumulate_forecasts_precip_accum when accum_hours > step_hours. """

    def test_second_call_returns_cached_without_recomputing(self: "TestFcstAccumPrecipCacheHit") -> None:
        """
        This test verifies that when accumulate_forecasts_precip_accum is called a second time with the same valid time and initialization string, it returns the cached accumulated DataArray from the first call without invoking the accumulation logic again. By stubbing the accumulate_forecasts method to track calls and returning a known DataArray, we can assert that the first call performs the expected number of accumulations and that the second call retrieves the result from the in-memory cache without additional calls to accumulate_forecasts, confirming that the caching mechanism is functioning correctly. 

        Parameters:
            self: the test-class instance.

        Returns:
            None
        """
        cfg = ModvxConfig(
            base_dir="/work",
            mpas_grid_file="grid/x1.nc",
            forecast_step_hours=1,
            precip_accum_hours=3,
        )

        fm_obj = FileManager(cfg)
        fm_obj.config.cache_dir = None

        mock_da = xr.DataArray(np.ones((3, 3)), dims=["y", "x"])
        vt = datetime.datetime(2024, 9, 17, 6)
        init_string = "2024091700"

        # Stub accumulate_forecasts so the first call can run without file I/O.
        accumulate_calls: list = []

        def _fake_accumulate(valid_time: datetime.datetime, 
                             init_str: str) -> xr.DataArray:
            """
            This stub function simulates the behavior of accumulate_forecasts by logging its input arguments and returning a predefined xarray DataArray. By appending the received valid time to an accumulate_calls list, we can later verify that the method was called with the expected sequence of valid times corresponding to the multi-step accumulation. The returned mock DataArray allows us to confirm that accumulate_forecasts_precip_accum correctly sums the results from multiple calls to accumulate_forecasts when the accumulation hours exceed the step hours, while also enabling us to track how many times the stub was invoked across multiple calls to accumulate_forecasts_precip_accum.

            Parameters:
                valid_time (datetime.datetime): The valid time argument passed to the stub.
                init_str (str): The initialization string argument passed to the stub.

            Returns:
                xr.DataArray: A predefined mock DataArray to simulate the output of accumulate_forecasts. 
            """
            accumulate_calls.append(valid_time)
            return mock_da

        fm_obj.accumulate_forecasts = _fake_accumulate

        # First call: computes the multi-step sum and caches it.
        result1 = fm_obj.accumulate_forecasts_precip_accum(vt, init_string)
        calls_after_first = len(accumulate_calls)

        # Reset stub call counter.
        accumulate_calls.clear()

        # Second call: must return the cached value without calling the stub.
        result2 = fm_obj.accumulate_forecasts_precip_accum(vt, init_string)

        # First call returns the *sum* of 3 all-ones slices (3.0 everywhere).
        assert float(result1.values.mean()) == pytest.approx(3.0)
        # Second call must return the exact same cached object.
        assert result2 is result1
        # Cache hit: accumulate_forecasts must NOT have been called again.
        assert len(accumulate_calls) == 0
        # The first call required N=3 step invocations (1h step, 3h accum).
        assert calls_after_first == 3


class TestObsAccumPrecipCacheHit:
    """ Tests for in-memory cache hit in accumulate_observations_precip_accum when accum_hours > obs_interval_hours. """

    def test_returns_cached_array_immediately(self: "TestObsAccumPrecipCacheHit") -> None:
        """
        This test verifies that when accumulate_observations_precip_accum is called with a valid time for which the accumulated result is already present in the in-memory cache, it returns the cached xarray DataArray immediately without performing any file I/O or computation. By pre-populating the _obs_mem_cache with a known DataArray under the expected cache key, we can assert that a call to accumulate_observations_precip_accum with the corresponding valid time retrieves this cached object directly, confirming that the cache hit logic is functioning correctly.

        Parameters:
            self: the test-class instance.

        Returns:
            None
        """
        cfg = ModvxConfig(
            base_dir="/work",
            mpas_grid_file="grid/x1.nc",
            forecast_step_hours=1,
            observation_interval_hours=1,
            precip_accum_hours=3,
        )
        fm_obj = FileManager(cfg)

        mock_da = xr.DataArray(np.ones((3, 3)), dims=["y", "x"])
        vt = datetime.datetime(2024, 9, 17, 0)
        accum_h = cfg.effective_precip_accum_hours  # 3

        # Construct the cache key exactly as the method does (line 589).
        key = f"obs_accum_{vt.strftime('%Y%m%d%H')}_{accum_h}h"
        fm_obj._obs_mem_cache[key] = mock_da

        result = fm_obj.accumulate_observations_precip_accum(vt)

        assert result is mock_da


class TestObsMultiStepAccumulation:
    """ Tests for multi-step accumulation in accumulate_observations_precip_accum when accum_hours > obs_interval_hours. """

    def test_sums_correct_number_of_hourly_slices(self: "TestObsMultiStepAccumulation",
                                                  tmp_path: str,) -> None:
        """
        This test verifies that accumulate_observations_precip_accum correctly sums the expected number of hourly slices from the daily FIMERG file when the configured precipitation accumulation hours exceed the observation interval hours. By creating a synthetic daily observation file with known values and stubbing the get_observation_filepath method to return this file, we can assert that the method retrieves and sums the correct number of hourly slices (based on the accumulation configuration) and produces an accumulated result with the expected numerical value.

        Parameters:
            tmp_path: pytest temporary directory.

        Returns:
            None
        """
        obs_var = "precip"
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            mpas_grid_file="grid/x1.nc",
            forecast_step_hours=1,
            observation_interval_hours=1,
            precip_accum_hours=3,
            obs_var_name=obs_var,
        )

        fm_obj = FileManager(cfg)
        fm_obj._obs_mem_cache.clear()
        fm_obj.config.cache_dir = None

        # Create a minimal daily obs file: 24 time steps × 3 × 3 lat/lon.
        data = np.ones((24, 3, 3), dtype=np.float32)
        ds = xr.Dataset({obs_var: xr.DataArray(data, dims=["time", "lat", "lon"])})
        obs_file = tmp_path / "obs_20240917.nc"
        ds.to_netcdf(str(obs_file))

        # Stub get_observation_filepath so every date key resolves to our file.
        fm_obj.get_observation_filepath = lambda date_key: str(obs_file)

        vt = datetime.datetime(2024, 9, 17, 0)
        result = fm_obj.accumulate_observations_precip_accum(vt)

        # 3 slices of all-1.0 → mean == 3.0
        assert float(result.values.mean()) == pytest.approx(3.0)

    def test_result_is_cached_after_computation(self: "TestObsMultiStepAccumulation",
                                                tmp_path: str,) -> None:
        """
        This test verifies that after accumulate_observations_precip_accum completes the multi-step accumulation, the resulting accumulated DataArray is stored in the _obs_mem_cache under the correct key. By calling the method with a valid time and then checking the contents of the cache, we can confirm that the accumulated result is cached for future retrieval, which is essential for optimizing performance by avoiding redundant computations on subsequent calls with the same valid time.

        Parameters:
            tmp_path: pytest temporary directory.

        Returns:
            None
        """
        obs_var = "precip"
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            mpas_grid_file="grid/x1.nc",
            forecast_step_hours=1,
            observation_interval_hours=1,
            precip_accum_hours=3,
            obs_var_name=obs_var,
        )
        fm_obj = FileManager(cfg)
        fm_obj._obs_mem_cache.clear()
        fm_obj.config.cache_dir = None

        data = np.ones((24, 3, 3), dtype=np.float32)
        ds = xr.Dataset({obs_var: xr.DataArray(data, dims=["time", "lat", "lon"])})
        obs_file = tmp_path / "obs_20240917.nc"
        ds.to_netcdf(str(obs_file))

        file_open_count: list = []

        def _counting_get_path(date_key: str) -> str:
            """
            This stub function replaces the get_observation_filepath method to count how many times it is called with each date key. By appending the received date_key to a file_open_count list, we can later verify that the method is called the expected number of times during the multi-step accumulation process. This allows us to confirm that the first call to accumulate_observations_precip_accum performs the necessary file I/O to read the observation data, while subsequent calls with the same valid time hit the cache without invoking this method again.

            Parameters:
                date_key (str): The date key for which the observation file path is requested.  

            Returns:
                str: The file path to the observation file corresponding to the given date key. 
            """
            file_open_count.append(date_key)
            return str(obs_file)

        fm_obj.get_observation_filepath = _counting_get_path

        vt = datetime.datetime(2024, 9, 17, 0)
        # First call: fills the cache.
        fm_obj.accumulate_observations_precip_accum(vt)
        calls_after_first = len(file_open_count)

        # Second call: must hit the cache without opening any file.
        file_open_count.clear()
        fm_obj.accumulate_observations_precip_accum(vt)

        assert calls_after_first >= 1        # first call did I/O
        assert len(file_open_count) == 0     # second call did not


class TestExtractFssToCsvExceptionHandling:
    """ Tests for exception handling in extract_fss_to_csv, ensuring that parse errors are swallowed and do not prevent CSV generation for other files. """

    def test_parse_error_is_swallowed_and_no_csv_written(self: "TestExtractFssToCsvExceptionHandling",
                                                         tmp_path: str,) -> None:
        """
        This test verifies that if _parse_records_from_nc_file raises an exception (e.g., due to a malformed .nc file), the extract_fss_to_csv method catches this exception internally and does not write any CSV output for that file. By creating a fake .nc file that triggers a parse error and then calling extract_fss_to_csv, we can assert that no CSV files are generated in the output directory, confirming that the method handles parse errors gracefully without crashing or producing invalid output.

        Parameters:
            tmp_path (str): pytest temporary directory.

        Returns:
            None
        """
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            experiment_name="test_exp",
            output_dir="output",
            csv_dir="csv",
        )
        fm_obj = FileManager(cfg)

        # Create a fake .nc file that the glob will discover.
        nc_dir = tmp_path / "output" / "ExtendedFC" / "2024091700" / "pp3h"
        nc_dir.mkdir(parents=True)
        fake_nc = nc_dir / "metrics.nc"
        fake_nc.write_text("not a valid netcdf")

        csv_dir = tmp_path / "csv"

        # Stub _parse_records_from_nc_file to simulate a parse failure.
        def _always_raise(nc_file: str) -> tuple[str, list[dict]]:
            """
            This stub function simulates a parse failure by always raising a RuntimeError when called. By replacing the _parse_records_from_nc_file method with this stub, we can test the exception handling logic in extract_fss_to_csv to ensure that it catches the exception and prevents it from propagating, which would otherwise disrupt the CSV generation process. This allows us to confirm that even when a parse error occurs, the method continues to execute without crashing and does not produce any CSV output for the problematic file. 

            Parameters:
                nc_file (str): The path to the .nc file being parsed.

            Returns:
                tuple[str, list[dict]]: This function does not return normally; it always raises an exception.
            """
            raise RuntimeError("simulated parse error")

        fm_obj._parse_records_from_nc_file = _always_raise

        # Must NOT raise — the exception must be caught and logged internally.
        fm_obj.extract_fss_to_csv(
            output_dir=str(tmp_path / "output"),
            csv_dir=str(csv_dir),
        )

        # No records were parsed, so no CSV should have been written.
        csv_files = list(csv_dir.glob("*.csv")) if csv_dir.exists() else []
        assert csv_files == []

    def test_partial_failure_still_writes_good_records(self: "TestExtractFssToCsvExceptionHandling",
                                                       tmp_path: str,) -> None:
        """
        This test verifies that if _parse_records_from_nc_file raises an exception for one .nc file but successfully parses another .nc file, the extract_fss_to_csv method still generates a CSV file containing the successfully parsed records. By creating two fake .nc files—one that triggers a parse error and another that returns valid records—and then calling extract_fss_to_csv, we can assert that a CSV file is generated with the correct content for the good file, confirming that the method continues processing other files even when some fail to parse. 

        Parameters:
            tmp_path (str): pytest temporary directory.

        Returns:
            None
        """
        import pandas as pd

        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            experiment_name="exp1",
            output_dir="output",
            csv_dir="csv",
        )
        fm_obj = FileManager(cfg)

        nc_dir = tmp_path / "output" / "ExtendedFC" / "2024091700" / "pp3h"
        nc_dir.mkdir(parents=True)
        bad_nc  = nc_dir / "bad.nc"
        good_nc = nc_dir / "good.nc"
        bad_nc.write_text("not a valid netcdf")
        good_nc.write_text("not a valid netcdf")

        csv_dir = tmp_path / "csv"

        # Stub: raise for the bad file, return a synthetic record for the good file.
        _good_record = {
            "initTime": "2024091700",
            "leadTime": 3,
            "domain": "GLOBAL",
            "thresh": 90.0,
            "window": 3,
            "fss": 0.75,
        }

        def _selective_parse(nc_file: str) -> tuple[str, list[dict]]:
            """ 
            This stub simulates the behavior of _parse_records_from_nc_file by raising an exception for a specific "bad.nc" file while returning a valid record for any other file. This allows us to test the exception handling logic in extract_fss_to_csv, ensuring that it can gracefully handle parse errors without affecting the processing of other files.
            
            Parameters:
                nc_file: The path to the .nc file being parsed.

            Returns:
                A tuple containing the experiment name and a list of parsed records.
            """
            if "bad.nc" in nc_file:
                raise RuntimeError("simulated parse error")
            return ("exp1", [_good_record])

        fm_obj._parse_records_from_nc_file = _selective_parse

        fm_obj.extract_fss_to_csv(
            output_dir=str(tmp_path / "output"),
            csv_dir=str(csv_dir),
        )

        csv_file = csv_dir / "exp1.csv"
        assert csv_file.exists(), "CSV for the good experiment must exist"
        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert df.iloc[0]["domain"] == "GLOBAL"
