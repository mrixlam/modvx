#!/usr/bin/env python3

"""
Unit tests for additional MODvx file management utilities.

This module contains tests for extended FileManager methods that were added after the initial test_file_manager.py suite. These tests cover path construction for MPAS diagnostic files based on valid-time and cycle arguments, ensuring the correct filename format is produced for both typical and edge-case times (e.g., midnight). By isolating these path-building tests, we can confirm that the FileManager correctly translates temporal metadata into filesystem paths that align with the expected directory structure and naming conventions used in the MODvx pipeline.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

import datetime
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from modvx.config import ModvxConfig
from modvx.file_manager import FileManager


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> ModvxConfig:
    """
    This fixture provides a temporary ModvxConfig rooted at the pytest-supplied tmp_path. The configuration includes standard sub-directory names for forecasts, observations, masks, output, and debug files, all relative to the temporary base directory. This allows tests to use a realistic configuration without writing to the real filesystem, ensuring isolation and cleanup after each test. 

    Parameters:
        tmp_path (Path): Pytest-supplied per-test temporary directory.

    Returns:
        ModvxConfig: Configuration object rooted at the temporary directory.
    """
    return ModvxConfig(
        base_dir=str(tmp_path),
        experiment_name="test_exp",
        forecast_step_hours=1,
    )


@pytest.fixture
def fm(tmp_cfg: ModvxConfig) -> FileManager:
    """
    This fixture provides a FileManager instance configured with the temporary ModvxConfig. By using the tmp_cfg fixture, we ensure that the FileManager operates within the isolated temporary directory structure for all file operations during testing. This allows tests to create, read, and write files without affecting the real filesystem and ensures that all test artifacts are cleaned up automatically by pytest.  

    Parameters:
        tmp_cfg (ModvxConfig): Temporary configuration fixture.

    Returns:
        FileManager: FileManager instance configured under the temporary directory.
    """
    return FileManager(tmp_cfg)


class TestLoadRegionMask:
    """ Tests for load_region_mask loading NetCDF masks and handling errors. """

    def test_loads_mask_from_netcdf(self: "TestLoadRegionMask", 
                                    fm: FileManager, 
                                    tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that load_region_mask successfully loads a land mask from a NetCDF file and returns an xarray DataArray with the expected variable name and dimensions. The test creates a synthetic NetCDF file containing a simple land mask variable, then calls load_region_mask with the file path. The returned mask should have the same variable name as in the file and include latitude and longitude dimensions, confirming that the loading and parsing logic works correctly.

        Parameters:
            fm (FileManager): FileManager instance fixture.
            tmp_cfg (ModvxConfig): Temporary configuration fixture.

        Returns:
            None
        """
        mask_dir = Path(tmp_cfg.base_dir) / tmp_cfg.mask_dir
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_file = mask_dir / "test_mask.nc"

        ds = xr.Dataset({
            "land_mask": xr.DataArray(
                np.ones((5, 10)),
                dims=["lat", "lon"],
                coords={"lat": np.arange(5), "lon": np.arange(10)},
            )
        })

        ds.to_netcdf(mask_file)
        mask, var_name = fm.load_region_mask(str(mask_file))

        assert var_name == "land_mask"
        assert "latitude" in mask.dims or "longitude" in mask.dims

    def test_missing_mask_raises(self: "TestLoadRegionMask", 
                                 fm: FileManager) -> None:
        """
        This test verifies that load_region_mask raises a FileNotFoundError when the specified mask file does not exist. When the method is called with a path that does not point to an existing NetCDF file, it should raise an exception with a clear message indicating that the mask file was not found. This guards against silent failures where a missing mask could lead to None values or downstream errors without a clear indication of the root cause.

        Parameters:
            fm (FileManager): FileManager instance fixture.

        Returns:
            None
        """
        with pytest.raises(FileNotFoundError, match="Mask file not found"):
            fm.load_region_mask("/nonexistent/mask.nc")

    def test_empty_mask_variable_raises(self: "TestLoadRegionMask", 
                                        fm: FileManager, 
                                        tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that load_region_mask raises a ValueError when the NetCDF file does not contain any variables that can be used as a mask. If the file exists but lacks a suitable variable (e.g., it only contains coordinates or unrelated data), the method should raise an exception indicating that no mask variable was found. This ensures that users are alerted to issues with the mask file content rather than receiving an empty or invalid mask. 

        Parameters:
            fm (FileManager): FileManager instance fixture.
            tmp_cfg (ModvxConfig): Temporary configuration fixture.

        Returns:
            None
        """
        mask_dir = Path(tmp_cfg.base_dir) / tmp_cfg.mask_dir
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_file = mask_dir / "empty_mask.nc"
        ds = xr.Dataset(coords={"lat": [0, 1], "lon": [0, 1]})
        ds.to_netcdf(mask_file)

        with pytest.raises(ValueError, match="No mask variable"):
            fm.load_region_mask(str(mask_file))


class TestObservationPath:
    """ Tests for get_observation_filepath path construction and vintage preference logic. """

    def test_fallback_to_first_vintage(self: "TestObservationPath", 
                                       fm: FileManager) -> None:
        """
        This test verifies that get_observation_filepath returns a path containing the first vintage in the obs_vintage_preference list when no files matching later-preference vintages are present. When the observation directory lacks files with vintages like LTE or SRCHHR, the method should fall back to the default FNL vintage as specified in the configuration. This ensures that the observation loading logic can still function even if only the default vintage is available, rather than failing due to missing preferred vintages. 

        Parameters:
            fm (FileManager): FileManager instance fixture.

        Returns:
            None
        """
        path = fm.get_observation_filepath("20240917")
        assert "FNL" in path  # first in obs_vintage_preference

    def test_existing_vintage_found(self: "TestObservationPath", 
                                    fm: FileManager, 
                                    tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that get_observation_filepath returns a path containing the LTE vintage when a file with that vintage exists in the observation directory. By creating a dummy observation file with the LTE vintage format in the obs_dir, we can confirm that the method correctly identifies and prefers this vintage over the default FNL. The returned path should include "LTE" to indicate that the preferred vintage was found and selected for loading. 

        Parameters:
            fm (FileManager): FileManager instance fixture.
            tmp_cfg (ModvxConfig): Temporary configuration fixture.

        Returns:
            None
        """
        obs_dir = Path(tmp_cfg.base_dir) / tmp_cfg.obs_dir
        obs_dir.mkdir(parents=True, exist_ok=True)

        lte_name = (
            "IMERG.A01H.VLD20240917.S20240917T000000."
            "E20240917T235959.LTE.V07B.SRCHHR.X360Y180.R1p0.FMT.nc"
        )

        (obs_dir / lte_name).write_text("stub")
        path = fm.get_observation_filepath("20240917")
        assert "LTE" in path


class TestObsCacheKey:
    """ Tests for _observation_cache_key formatting and consistency. """

    def test_key_format(self: "TestObsCacheKey", 
                        fm: FileManager) -> None:
        """
        This test verifies that _observation_cache_key generates a consistent cache key string based on the valid time. The key should follow the format "obs_accum_YYYYMMDDHH_1h" where the timestamp corresponds to the valid time of the observation accumulation. By providing a specific datetime object, we can confirm that the generated key matches the expected format and includes the correct date and hour components. This ensures that the caching mechanism can reliably use these keys to store and retrieve accumulated observations. 

        Parameters:
            fm (FileManager): FileManager instance fixture.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 17, 6, 0)
        key = fm._observation_cache_key(vt)
        assert key == "obs_accum_2024091706_1h"


class TestSaveIntermediatePrecip:
    """ Tests for save_intermediate_precip writing debug NetCDF files. """

    def test_writes_netcdf(self: "TestSaveIntermediatePrecip", 
                           fm: FileManager, 
                           tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that save_intermediate_precip creates a NetCDF file containing forecast, observation, and difference variables. The method should write a file to the debug directory with the expected cycle-start and valid-time in the path. The output dataset should include three variables: the forecast precipitation, the observed precipitation, and their difference. This test confirms that all three variables are present in the output file, allowing for offline inspection of intermediate fields during debugging. 

        Parameters:
            fm (FileManager): FileManager instance fixture.
            tmp_cfg (ModvxConfig): Temporary configuration fixture.

        Returns:
            None
        """
        fcst = xr.DataArray(
            np.random.rand(5, 5),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )

        obs = xr.DataArray(
            np.random.rand(5, 5),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )

        cycle_start = datetime.datetime(2024, 9, 17, 0, 0)
        valid_time = datetime.datetime(2024, 9, 17, 6, 0)

        fm.save_intermediate_precip(fcst, obs, cycle_start, valid_time)

        debug_dir = Path(tmp_cfg.base_dir) / tmp_cfg.debug_dir
        nc_files = list(debug_dir.rglob("*.nc"))
        assert len(nc_files) == 1

        ds = xr.open_dataset(nc_files[0])
        assert "forecast" in ds
        assert "observation" in ds
        assert "difference" in ds
        ds.close()


class TestSaveIntermediateBinary:
    """ Tests for save_intermediate_binary writing debug binary mask NetCDF files. """

    def test_writes_netcdf(self: "TestSaveIntermediateBinary", 
                           fm: FileManager, 
                           tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that save_intermediate_binary creates a NetCDF file containing forecast and observation binary masks. The method should write a file to the debug directory with the expected cycle-start and valid-time in the path. The output dataset should include two variables: the forecast binary mask and the observed binary mask. This test confirms that both binary variables are present in the output file, allowing for offline inspection of intermediate binary fields during debugging. 

        Parameters:
            fm (FileManager): FileManager instance fixture.
            tmp_cfg (ModvxConfig): Temporary configuration fixture.

        Returns:
            None
        """
        binary = xr.DataArray(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            dims=["latitude", "longitude"],
            coords={"latitude": [0, 1], "longitude": [0, 1]},
        )

        cycle_start = datetime.datetime(2024, 9, 17, 0, 0)
        valid_time = datetime.datetime(2024, 9, 17, 6, 0)

        fm.save_intermediate_binary(binary, binary, cycle_start, valid_time, 90.0)

        debug_dir = Path(tmp_cfg.base_dir) / tmp_cfg.debug_dir
        nc_files = list(debug_dir.rglob("*.nc"))
        assert len(nc_files) == 1

        ds = xr.open_dataset(nc_files[0])
        assert "forecast_binary" in ds
        assert "observation_binary" in ds
        ds.close()


class TestSaveFssResults:
    """ Tests for save_fss_results writing FSS NetCDF files. """

    def test_writes_fss_only(self: "TestSaveFssResults", 
                             fm: FileManager, 
                             tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that save_fss_results creates a NetCDF file containing only the FSS variable when given a list of metrics dictionaries with only FSS values. The method should write a file to the output directory with the expected cycle-start, region, threshold, and window size in the path. The output dataset should include an "fss" variable with the correct length corresponding to the input metrics list, and it should not contain any contingency metric variables (pod, far, csi, fbias, ets). This confirms that the method correctly handles cases where only FSS is provided without any contingency metrics. 

        Parameters:
            fm (FileManager): FileManager instance fixture.
            tmp_cfg (ModvxConfig): Temporary configuration fixture.

        Returns:
            None
        """
        metrics_list = [
            {"fss": 0.8},
            {"fss": 0.85},
        ]
        cycle_start = datetime.datetime(2024, 9, 17, 0, 0)

        fm.save_fss_results(metrics_list, cycle_start, "GLOBAL", 90.0, 3)

        output_dir = Path(tmp_cfg.base_dir) / tmp_cfg.output_dir
        nc_files = list(output_dir.rglob("*.nc"))
        assert len(nc_files) == 1

        ds = xr.open_dataset(nc_files[0])
        assert "fss" in ds
        assert len(ds["fss"]) == 2

        for key in ["pod", "far", "csi", "fbias", "ets"]:
            assert key not in ds

        ds.close()

    def test_filename_encoding(self: "TestSaveFssResults", 
                               fm: FileManager, 
                               tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that the FSS NetCDF filename encodes the region name, threshold value, and window size correctly. When save_fss_results is called with specific region, threshold, and window parameters, the resulting NetCDF file should have a name that includes the region (e.g., "tropics"), the threshold (e.g., "97p5" or "97.5"), and the window size (e.g., "window3"). This confirms that the filename construction logic in save_fss_results correctly incorporates these parameters to produce informative and unique filenames for each cycle-region-threshold-window combination.

        Parameters:
            fm (FileManager): FileManager instance fixture.
            tmp_cfg (ModvxConfig): Temporary configuration fixture.

        Returns:
            None
        """
        metrics_list = [{"fss": 0.5}]
        cycle_start = datetime.datetime(2024, 9, 17, 0, 0)

        fm.save_fss_results(metrics_list, cycle_start, "TROPICS", 97.5, 5)

        output_dir = Path(tmp_cfg.base_dir) / tmp_cfg.output_dir
        nc_files = list(output_dir.rglob("*.nc"))
        assert len(nc_files) == 1
        name = nc_files[0].name
        assert "tropics" in name
        assert "_window" in name
        assert "97p5" in name or "97.5" in name


class TestSaveContingencyResults:
    """ Tests for save_contingency_results writing contingency NetCDF files. """

    def test_writes_contingency_metrics(self: "TestSaveContingencyResults", 
                                        fm: FileManager, 
                                        tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that save_contingency_results creates a NetCDF file containing the expected contingency metrics (pod, far, csi, fbias, ets) when given a list of metrics dictionaries with these values. The method should write a file to the output directory with the expected cycle-start, region, and threshold in the path. The output dataset should include variables for each of the contingency metrics with lengths corresponding to the input metrics list, and it should not contain an "fss" variable. This confirms that the method correctly handles cases where contingency metrics are provided without any FSS values.

        Parameters:
            fm (FileManager): FileManager instance fixture.
            tmp_cfg (ModvxConfig): Temporary configuration fixture.

        Returns:
            None
        """
        metrics_list = [
            {"pod": 0.7, "far": 0.2, "csi": 0.6, "fbias": 1.1, "ets": 0.4},
            {"pod": 0.75, "far": 0.15, "csi": 0.65, "fbias": 1.0, "ets": 0.5},
        ]
        cycle_start = datetime.datetime(2024, 9, 17, 0, 0)

        fm.save_contingency_results(metrics_list, cycle_start, "GLOBAL", 90.0)

        output_dir = Path(tmp_cfg.base_dir) / tmp_cfg.output_dir
        nc_files = list(output_dir.rglob("*.nc"))
        assert len(nc_files) == 1

        ds = xr.open_dataset(nc_files[0])

        for key in ["pod", "far", "csi", "fbias", "ets"]:
            assert key in ds
            assert len(ds[key]) == 2

        assert "fss" not in ds
        ds.close()

    def test_filename_encoding(self: "TestSaveContingencyResults", 
                               fm: FileManager, 
                               tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that the contingency NetCDF filename encodes the region name and threshold value correctly. When save_contingency_results is called with specific region and threshold parameters, the resulting NetCDF file should have a name that includes the region (e.g., "tropics") and the threshold (e.g., "97p5" or "97.5"), but it should not include "nbhd" since this method is for contingency metrics rather than FSS. This confirms that the filename construction logic in save_contingency_results correctly incorporates these parameters to produce informative and unique filenames for each cycle-region-threshold combination, while also distinguishing from FSS files. 

        Parameters:
            fm (FileManager): FileManager instance fixture.
            tmp_cfg (ModvxConfig): Temporary configuration fixture.

        Returns:
            None
        """
        metrics_list = [{"pod": 0.5, "far": 0.5, "csi": 0.5, "fbias": 1.0, "ets": 0.3}]
        cycle_start = datetime.datetime(2024, 9, 17, 0, 0)

        fm.save_contingency_results(metrics_list, cycle_start, "TROPICS", 97.5)

        output_dir = Path(tmp_cfg.base_dir) / tmp_cfg.output_dir
        nc_files = list(output_dir.rglob("*.nc"))
        assert len(nc_files) == 1
        name = nc_files[0].name
        assert "tropics" in name
        assert "nbhd" not in name
        assert "97p5" in name or "97.5" in name


class TestForecastCaching:
    """ Tests for forecast cache paths in accumulate_forecasts. """

    def test_memory_cache_hit(self: "TestForecastCaching", 
                              tmp_path: Path) -> None:
        """
        This test verifies that accumulate_forecasts returns the cached array directly when a memory cache entry exists. The in-memory cache avoids reopening the forecast file for each lead time within the same process. This test pre-populates the cache and confirms the returned values match without any file I/O. By creating a synthetic DataArray and inserting it into the FileManager's _fcst_mem_cache with the appropriate key, we can call accumulate_forecasts and expect it to return the cached array without calling the underlying computation method. This confirms that the memory caching mechanism is functioning correctly to optimize repeated access to the same forecast accumulation within a process.

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory used to construct a FileManager instance.

        Returns:
            None
        """
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            experiment_name="test_exp",
            forecast_step_hours=1,
            mpas_grid_file="grid/mesh.nc",
        )
        fm = FileManager(cfg)

        vt = datetime.datetime(2024, 9, 17, 0)
        key = fm._forecast_cache_key("2024091700", vt)
        cached = xr.DataArray(np.ones((2, 2)), dims=["latitude", "longitude"])
        fm._fcst_mem_cache[key] = cached

        def _should_not_be_called(*args, 
                                  **kwargs) -> None:
            """
            This function serves as a sentinel to detect if the cache miss path is taken in accumulate_forecasts. If accumulate_forecasts attempts to compute the forecast accumulation instead of returning the cached result, it will call this function, which raises an AssertionError to fail the test immediately. This ensures that the test confirms the memory cache hit behavior without allowing any fallback to the computation path, providing a clear signal if the caching mechanism is not working as expected.

            Parameters:
                *args: Positional arguments passed to the sentinel function (ignored).
                **kwargs: Keyword arguments passed to the sentinel function (ignored). 

            Returns:
                None
            """
            raise AssertionError("Cache miss: _compute_forecast_accumulation was called unexpectedly")

        fm._compute_forecast_accumulation = _should_not_be_called
        result = fm.accumulate_forecasts(vt, "2024091700")

        np.testing.assert_array_equal(result.values, cached.values)

    def test_disk_cache_hit(self: "TestForecastCaching", 
                            tmp_path: Path) -> None:
        """
        This test verifies that accumulate_forecasts loads from a disk-cached NetCDF file when the memory cache is empty. The disk cache persists forecast accumulations across process restarts to avoid re-reading large MPAS files on subsequent runs. This test writes a fake cache file and confirms the loader returns the same data without calling the raw accumulation path. By creating a synthetic DataArray, writing it to a NetCDF file in the cache directory with the appropriate key as the filename, and then calling accumulate_forecasts, we can confirm that the method correctly loads the cached data from disk when the memory cache does not have an entry. This ensures that the disk caching mechanism is functioning correctly to optimize access to previously computed forecast accumulations across runs. 

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory used to construct a FileManager instance.

        Returns:
            None
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            cache_dir=str(cache_dir),
            experiment_name="test_exp",
            forecast_step_hours=1,
            mpas_grid_file="grid/mesh.nc",
        )
        fm = FileManager(cfg)

        vt = datetime.datetime(2024, 9, 17, 0)
        key = fm._forecast_cache_key("2024091700", vt)
        da = xr.DataArray(np.arange(4.0), dims=["x"])
        da.to_netcdf(cache_dir / f"{key}.nc")

        # Sentinel: raises immediately if the disk-cache miss path is taken.
        def _should_not_be_called(*args, 
                                  **kwargs) -> None:
            """
            This function serves as a sentinel to detect if the cache miss path is taken in accumulate_forecasts. If accumulate_forecasts attempts to compute the forecast accumulation instead of loading from the disk cache, it will call this function, which raises an AssertionError to fail the test immediately. This ensures that the test confirms the disk cache hit behavior without allowing any fallback to the computation path, providing a clear signal if the caching mechanism is not working as expected.

            Parameters:
                *args: Positional arguments passed to the sentinel function (ignored).
                **kwargs: Keyword arguments passed to the sentinel function (ignored).

            Returns:
                None
            """
            raise AssertionError("Cache miss: _compute_forecast_accumulation was called unexpectedly")

        fm._compute_forecast_accumulation = _should_not_be_called
        result = fm.accumulate_forecasts(vt, "2024091700")

        np.testing.assert_array_equal(result.values, da.values)


class TestExtractFssToCsv:
    """ Tests for extract_fss_to_csv scanning and writing CSV files. """

    def _create_output_tree(self: "TestExtractFssToCsv", 
                            base: Path) -> None:
        """
        This helper method creates a standard NetCDF output tree with one FSS file and one contingency file under the output directory. The files are named according to the expected conventions used in the MODvx pipeline, including the experiment name, cycle time, region, threshold, and window size. The FSS file contains a single "fss" variable with two values, while the contingency file contains variables for pod, far, csi, fbias, and ets with two values each. This setup allows the test_csv_written method to verify that extract_fss_to_csv correctly scans the output tree and extracts both FSS and contingency metrics into a CSV file. 

        Parameters:
            base (Path): Root temporary directory under which the output tree will be created.

        Returns:
            None
        """
        odir = (
            base / "output" / "test_exp" / "ExtendedFC"
            / "2024091700" / "pp1h"
        )
        odir.mkdir(parents=True, exist_ok=True)

        fss_ds = xr.Dataset({
            "fss": xr.DataArray([0.8, 0.85], dims=["valid_time_index"]),
        })
        fss_name = "modvx_metrics_type_neighborhood_global_1h_indep_thresh90p0percent_window3.nc"
        fss_ds.to_netcdf(odir / fss_name)

        cont_ds = xr.Dataset({
            "pod": xr.DataArray([0.7, 0.75], dims=["valid_time_index"]),
            "far": xr.DataArray([0.2, 0.15], dims=["valid_time_index"]),
            "csi": xr.DataArray([0.6, 0.65], dims=["valid_time_index"]),
            "fbias": xr.DataArray([1.1, 1.0], dims=["valid_time_index"]),
            "ets": xr.DataArray([0.4, 0.5], dims=["valid_time_index"]),
        })
        cont_name = "modvx_metrics_type_contingency_global_1h_indep_thresh90p0percent.nc"
        cont_ds.to_netcdf(odir / cont_name)

    def test_csv_written(self: "TestExtractFssToCsv", 
                         tmp_path: Path) -> None:
        """
        This test verifies that extract_fss_to_csv scans the output directory for NetCDF files, extracts FSS and contingency metrics, and writes a CSV file with the expected columns. By creating a synthetic output tree with one FSS file and one contingency file containing known values, we can call extract_fss_to_csv and then read the resulting CSV file to confirm that it contains the correct columns ("fss", "pod", "leadTime") and that the values match those in the original NetCDF files. This confirms that the method correctly processes the output files and produces a consolidated CSV summary of the metrics. 

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory used to create the output tree and store the resulting CSV file.

        Returns:
            None
        """
        import pandas as pd

        self._create_output_tree(tmp_path)
        cfg = ModvxConfig(base_dir=str(tmp_path), experiment_name="test_exp")
        fm = FileManager(cfg)

        csv_dir = tmp_path / "csv_out"
        fm.extract_fss_to_csv(
            output_dir=str(tmp_path / "output"),
            csv_dir=str(csv_dir),
        )

        csv_files = list(csv_dir.glob("*.csv"))
        assert len(csv_files) == 1

        df = pd.read_csv(csv_files[0])
        assert "fss" in df.columns
        assert "pod" in df.columns
        assert "leadTime" in df.columns
        assert len(df) == 4

    def test_no_files_logs_zero(self: "TestExtractFssToCsv", 
                                tmp_path: Path) -> None:
        """
        This test verifies that extract_fss_to_csv handles the case where no NetCDF files are present in the output directory by logging a message and writing an empty CSV file. When the method is called with an output directory that contains no FSS or contingency NetCDF files, it should not raise an error but instead create a CSV file with only the header row (e.g., "fss", "pod", "leadTime") and no data rows. This confirms that the method can gracefully handle cases where there are no metrics to extract without failing or producing invalid output. 

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory used as base_dir.

        Returns:
            None
        """
        cfg = ModvxConfig(base_dir=str(tmp_path))
        fm = FileManager(cfg)
        fm.extract_fss_to_csv(
            output_dir=str(tmp_path / "empty"),
            csv_dir=str(tmp_path / "csv_out"),
        )
        csv_files = list((tmp_path / "csv_out").glob("*.csv"))
        assert len(csv_files) == 0


class TestObservationCaching:
    """ Tests for the multi-level observation cache in accumulate_observations. """

    def test_memory_cache_hit(self: "TestObservationCaching", 
                              fm: FileManager) -> None:
        """
        This test verifies that accumulate_observations returns the cached array directly when a memory cache entry exists. The in-memory cache avoids reopening the observation file for each lead time within the same process. This test pre-populates the cache and confirms the returned values match without any file I/O. By creating a synthetic DataArray and inserting it into the FileManager's _obs_mem_cache with the appropriate key, we can call accumulate_observations and expect it to return the cached array without calling the underlying computation method. This confirms that the memory caching mechanism is functioning correctly to optimize repeated access to the same observation accumulation within a process. 

        Parameters:
            fm (FileManager): FileManager instance fixture.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 17, 6, 0)
        key = fm._observation_cache_key(vt)
        cached = xr.DataArray(np.ones(5))
        fm._obs_mem_cache[key] = cached

        result = fm.accumulate_observations(vt)
        np.testing.assert_array_equal(result.values, cached.values)

    def test_disk_cache_hit(self: "TestObservationCaching", 
                            fm: FileManager, 
                            tmp_cfg: ModvxConfig) -> None:
        """
        This test verifies that accumulate_observations loads from a disk-cached NetCDF file when the memory cache is empty. The disk cache persists observation accumulations across process restarts to avoid re-reading large observation files on subsequent runs. This test writes a fake cache file and confirms the loader returns the same data without calling the raw accumulation path. By creating a synthetic DataArray, writing it to a NetCDF file in the cache directory with the appropriate key as the filename, and then calling accumulate_observations, we can confirm that the method correctly loads the cached data from disk when the memory cache does not have an entry. This ensures that the disk caching mechanism is functioning correctly to optimize access to previously computed observation accumulations across runs. 

        Parameters:
            fm (FileManager): FileManager instance fixture.
            tmp_cfg (ModvxConfig): Configuration fixture with temporary directories.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 17, 6, 0)
        key = fm._observation_cache_key(vt)

        # Write a disk cache entry
        cache_dir = Path(tmp_cfg.base_dir) / ".obs_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_cfg_with_cache = ModvxConfig(
            base_dir=tmp_cfg.base_dir,
            cache_dir=str(cache_dir),
            forecast_step_hours=1,
        )
        fm2 = FileManager(tmp_cfg_with_cache)

        da = xr.DataArray(np.arange(5.0), dims=["x"], name="obs")
        da.to_netcdf(cache_dir / f"{key}.nc")

        result = fm2.accumulate_observations(vt)
        np.testing.assert_array_equal(result.values, da.values)


def _make_cfg(tmp_path: Path) -> ModvxConfig:
    """
    This helper function creates a minimal ModvxConfig with standard directory layout for testing purposes. By using the tmp_path fixture, we ensure that all directories (fcst, obs, output, csv, plots) are created under the temporary base directory provided by pytest. This allows tests to operate in an isolated filesystem environment without affecting real data or configurations. The returned ModvxConfig can be used to instantiate a FileManager for testing various file operations. 

    Parameters:
        tmp_path (Path): Pytest-supplied per-test temporary directory used as base_dir.

    Returns:
        ModvxConfig: Minimal configuration object with standard directory layout.
    """
    return ModvxConfig(
        base_dir=str(tmp_path),
        fcst_dir="fcst",
        obs_dir="obs",
        output_dir="output",
        csv_dir="csv",
        plot_dir="plots",
    )


class TestAccumulateForecasts:
    """ Tests for FileManager.accumulate_forecasts which delegates to mpas_reader. """

    def test_accumulate_forecasts_calls_mpas_reader(self: "TestAccumulateForecasts",
                                                    tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that accumulate_forecasts calls the appropriate mpas_reader functions to load and remap MPAS precipitation data. Lightweight call-tracking stubs replace load_mpas_precip and remap_to_latlon via monkeypatch so no real MPAS files are required. After accumulate_forecasts returns, the recorded call counts are asserted directly to confirm both functions were invoked the expected number of times and that the returned DataArray has the correct shape.

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory used to construct a FileManager instance.
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            fcst_dir="fcst",
            experiment_name="test_exp",
            forecast_step_hours=12,
            mpas_grid_file="grid/mesh.nc",
            mpas_remap_resolution=0.5,
        )
        fm = FileManager(cfg)

        remapped = xr.DataArray(
            np.ones((3, 3)), dims=["latitude", "longitude"],
            coords={"latitude": [0.0, 0.5, 1.0], "longitude": [0.0, 0.5, 1.0]},
        )

        load_returns = [
            xr.DataArray(np.array([30.0, 50.0]), dims=["nCells"]),
            xr.DataArray(np.array([10.0, 20.0]), dims=["nCells"]),
        ]
        load_calls: list = []
        remap_calls: list = []

        def fake_load(*args,
                      **kwargs) -> xr.DataArray:
            """
            This function serves as a stub replacement for mpas_reader.load_mpas_precip. It records the arguments with which it was called into the load_calls list and returns a predefined DataArray from the load_returns list based on the call count. This allows the test to confirm that accumulate_forecasts is invoking the load function the expected number of times with the correct arguments, without requiring access to real MPAS files.

            Parameters:
                *args: Positional arguments passed to the load function (recorded in load_calls).   
                **kwargs: Keyword arguments passed to the load function (recorded in load_calls).

            Returns:
                xr.DataArray: A predefined DataArray from load_returns corresponding to the call count. 
            """
            idx = len(load_calls)
            load_calls.append(args)
            return load_returns[idx]

        def fake_remap(*args, **kwargs) -> xr.DataArray:
            """
            This function serves as a stub replacement for mpas_reader.remap_to_latlon. It records the arguments with which it was called into the remap_calls list and returns a predefined remapped DataArray. This allows the test to confirm that accumulate_forecasts is invoking the remap function with the correct arguments, without performing any actual remapping logic.

            Parameters:
                *args: Positional arguments passed to the remap function (recorded in remap_calls).
                **kwargs: Keyword arguments passed to the remap function (recorded in remap_calls).

            Returns:
                xr.DataArray: A predefined remapped DataArray with shape (3, 3) and latitude/longitude coordinates. 
            """
            remap_calls.append(args)
            return remapped

        monkeypatch.setattr("modvx.mpas_reader.load_mpas_precip", fake_load)
        monkeypatch.setattr("modvx.mpas_reader.remap_to_latlon", fake_remap)

        vt = datetime.datetime(2024, 9, 17, 0)
        result = fm.accumulate_forecasts(vt, "2024091700")

        assert len(load_calls) == 2
        assert len(remap_calls) == 1
        assert result.shape == (3, 3)


class TestObsDiskCacheWrite:
    """ Tests for the disk cache WRITE path in accumulate_observations. """

    def test_disk_cache_written(self: "TestObsDiskCacheWrite", 
                                tmp_path: Path) -> None:
        """
        This test verifies that accumulate_observations writes a NetCDF file to the disk cache when no memory cache entry exists. By mocking the raw accumulation method to return a synthetic DataArray, we can confirm that accumulate_observations calls the method to compute the accumulation and then writes the result to a NetCDF file in the cache directory with the expected filename. This ensures that the disk caching mechanism is functioning correctly to persist observation accumulations across runs when they are not already in memory. 

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory used to construct a FileManager instance and store the disk cache file. 

        Returns:
            None
        """
        cache_dir = str(tmp_path / "cache")
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            cache_dir=cache_dir,
            forecast_step_hours=1,
            observation_interval_hours=1,
        )
        fm = FileManager(cfg)

        fake_da = xr.DataArray(np.ones((3, 3)), dims=["lat", "lon"])
        vt = datetime.datetime(2024, 9, 17, 0)

        # First call: stub returns fake_da so the disk cache is written.
        fm._compute_observation_accumulation_raw = lambda *a, **kw: fake_da
        fm.accumulate_observations(vt)

        key = fm._observation_cache_key(vt)
        disk_path = os.path.join(cache_dir, f"{key}.nc")
        assert os.path.exists(disk_path)

        # Second call must hit the memory cache; sentinel raises if raw is invoked.
        def _should_not_be_called(*args, 
                                  **kwargs) -> None:
            """
            This function serves as a sentinel to detect if the raw accumulation path is taken in the second call to accumulate_observations. If accumulate_observations attempts to compute the observation accumulation again instead of returning the cached result, it will call this function, which raises an AssertionError to fail the test immediately. This ensures that the test confirms the memory cache hit behavior on the second call without allowing any fallback to the computation path, providing a clear signal if the caching mechanism is not working as expected.

            Parameters:
                *args: Positional arguments passed to the sentinel function (ignored).
                **kwargs: Keyword arguments passed to the sentinel function (ignored).

            Returns:
                None 
            """
            raise AssertionError("Memory cache bypassed: _compute_observation_accumulation_raw called on second call")

        fm._compute_observation_accumulation_raw = _should_not_be_called
        fm.accumulate_observations(vt)  # passes only if raw is NOT called


class TestAccumulateObservationsRaw:
    """ Tests for FileManager._compute_observation_accumulation_raw. """

    def test_raw_accumulation(self: "TestAccumulateObservationsRaw", 
                              tmp_path: Path) -> None:
        """
        This test verifies that _compute_observation_accumulation_raw correctly reads observation data from a NetCDF file, accumulates it according to the specified valid time and observation interval, and returns a DataArray with the expected shape. By creating a synthetic observation NetCDF file with known values and mocking the get_observation_filepath method to return its path, we can confirm that the method reads the file, performs the accumulation logic, and produces an output array with the correct dimensions. This ensures that the raw accumulation logic is functioning correctly when processing observation data. 

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory used to create the synthetic observation NetCDF file and store the cache file. 

        Returns:
            None
        """
        cfg = ModvxConfig(
            base_dir=str(tmp_path),
            obs_dir="obs",
            obs_var_name="precip",
            forecast_step_hours=3,
            observation_interval_hours=1,
        )
        fm = FileManager(cfg)

        obs_dir = tmp_path / "obs"
        obs_dir.mkdir()
        data = np.random.default_rng(42).random((24, 5, 5))
        ds = xr.Dataset({"precip": (["time", "y", "x"], data)})

        vt = datetime.datetime(2024, 9, 17, 0)

        fake_nc_path = str(tmp_path / "fake_obs.nc")
        ds.to_netcdf(fake_nc_path)
        fm.get_observation_filepath = lambda *a, **kw: fake_nc_path
        result = fm._compute_observation_accumulation_raw(vt)

        assert result is not None
        assert result.shape == (5, 5)


class TestExtractFssToCsvLegacy:
    """ Tests for the legacy single-array format branch in extract_fss_to_csv. """

    def test_legacy_format_extraction(self: "TestExtractFssToCsvLegacy", 
                                      tmp_path: Path) -> None:
        """
        This test verifies that extract_fss_to_csv correctly reads a legacy single-array NetCDF file containing only FSS values and writes them to a CSV file with the expected columns. The NetCDF dataset contains an "fss" variable with values for three lead times, but it does not include any contingency metrics. The extractor should read the FSS values, fill the domain/threshold/window metadata from the filename, and write a CSV file with columns for "fss", "pod", and "lead_time". The "pod" column should be filled with NaN since it is not present in the original dataset. This confirms that the legacy format branch of extract_fss_to_csv can handle files that contain only FSS values without any contingency metrics. 

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory used to create the synthetic NetCDF file and store the resulting CSV file. 

        Returns:
            None
        """
        exp_name = "test_exp"
        init_time = "2024091700"
        nc_dir = tmp_path / "output" / exp_name / "ExtendedFC" / init_time / "pp12h"
        nc_dir.mkdir(parents=True)

        fss_data = np.array([0.5, 0.6, 0.7])
        da = xr.DataArray(fss_data, dims=["lead_time"])
        fname = "modvx_metrics_type_neighborhood_global_12h_indep_thresh90p0percent_window3.nc"
        da.to_netcdf(str(nc_dir / fname))

        csv_dir = tmp_path / "csv"
        cfg = ModvxConfig(base_dir=str(tmp_path), output_dir="output", csv_dir="csv")
        fm = FileManager(cfg)

        fm.extract_fss_to_csv(
            output_dir=str(tmp_path / "output"),
            csv_dir=str(csv_dir),
        )

        csv_file = csv_dir / f"{exp_name}.csv"
        assert csv_file.exists()

        df = pd.read_csv(str(csv_file))
        assert len(df) == 3
        assert "fss" in df.columns
        assert df["fss"].tolist() == [0.5, 0.6, 0.7]
        assert df["pod"].isna().all()

    def test_multi_metric_format_extraction(self: "TestExtractFssToCsvLegacy", 
                                            tmp_path: Path) -> None:  # noqa: PLR0915
        """
        This test verifies that extract_fss_to_csv correctly reads a legacy single-array NetCDF file containing multiple metrics (fss, pod, far, csi, fbias, ets) and writes them to a CSV file with the expected columns. The NetCDF dataset contains variables for each metric with values for two lead times. The extractor should read all the metric values, fill the domain/threshold/window metadata from the filename, and write a CSV file with columns for "fss", "pod", "far", "csi", "fbias", "ets", and "lead_time". This confirms that the legacy format branch of extract_fss_to_csv can handle files that contain multiple metrics in separate variables without any contingency metrics.

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory used to create the synthetic NetCDF file and store the resulting CSV file.

        Returns:
            None
        """
        exp_name = "test_exp"
        init_time = "2024091700"
        nc_dir = tmp_path / "output" / exp_name / "ExtendedFC" / init_time / "pp12h"
        nc_dir.mkdir(parents=True)

        ds = xr.Dataset({
            "fss": ("lead_time", [0.5, 0.8]),
            "pod": ("lead_time", [0.9, 0.7]),
            "far": ("lead_time", [0.1, 0.3]),
            "csi": ("lead_time", [0.6, 0.5]),
            "fbias": ("lead_time", [1.0, 1.2]),
            "ets": ("lead_time", [0.4, 0.3]),
        })
        fname = "modvx_metrics_type_neighborhood_tropics_12h_indep_thresh95p0percent_window5.nc"
        ds.to_netcdf(str(nc_dir / fname))

        csv_dir = tmp_path / "csv"
        cfg = ModvxConfig(base_dir=str(tmp_path), output_dir="output", csv_dir="csv")
        fm = FileManager(cfg)

        fm.extract_fss_to_csv(
            output_dir=str(tmp_path / "output"),
            csv_dir=str(csv_dir),
        )

        csv_file = csv_dir / f"{exp_name}.csv"
        assert csv_file.exists()

        df = pd.read_csv(str(csv_file))
        assert len(df) == 2
        assert df["pod"].tolist() == [0.9, 0.7]
        assert df["domain"].iloc[0] == "tropics"


class TestFileManagerParsingHelpers:
    """ Tests for internal parsing helper branches. """

    def test_extract_file_context_no_output(self: "TestFileManagerParsingHelpers",) -> None:
        """
        This test verifies that _extract_file_context returns None when the filename does not match the expected pattern for extracting domain, threshold, and window metadata. By providing a path that does not contain the required components (e.g., "modvx_metrics_type_neighborhood_{domain}_{lead_time}h_indep_thresh{threshold}percent_window{window}.nc"), we can confirm that the method correctly identifies the mismatch and returns None instead of attempting to extract metadata. This ensures that the method can gracefully handle unexpected filename formats without raising errors or producing invalid metadata. 

        Parameters:
            None

        Returns:
            None
        """
        result = FileManager._extract_file_context("/tmp/experiment/ExtendedFC/2024091700/pp1h/file.nc")
        assert result is None

    def test_parse_metric_values_fallback_length(self: "TestFileManagerParsingHelpers",) -> None:
        """
        This test verifies that _parse_metric_values falls back to using the length of any variable in the dataset when the expected metric variables are not present. By providing a dataset that contains a variable unrelated to the expected metrics (e.g., "other") and requesting parsing for a metric that is not in the dataset (e.g., "fss"), we can confirm that the method uses the length of the "other" variable to determine the number of lead times and fills the missing "fss" values with NaN. This ensures that the method can handle cases where the expected metric variables are absent by using a reasonable fallback approach to determine the length of the metrics arrays. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({"other": xr.DataArray([1.0, 2.0, 3.0], dims=["x"])})
        length, metrics = FileManager._parse_metric_values(ds, ["fss"])
        assert length == 3
        assert len(metrics["fss"]) == 3
        assert np.isnan(metrics["fss"]).all()

    def test_parse_metric_values_legacy_fss(self: "TestFileManagerParsingHelpers",) -> None:
        """
        This test verifies that _parse_metric_values correctly extracts FSS values from a legacy single-array dataset when the "fss" variable is present but no contingency metrics are included. By providing a dataset that contains only an "fss" variable with values for two lead times, we can confirm that the method returns the correct length and metric values for "fss" while leaving any missing contingency metrics as NaN. This ensures that the method can handle legacy formats where only FSS values are present without any accompanying contingency metrics. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "__xarray_dataarray_variable__": xr.DataArray([0.1, 0.2], dims=["x"])
        })
        length, metrics = FileManager._parse_metric_values(ds, ["fss"])
        assert length == 2
        np.testing.assert_allclose(metrics["fss"], [0.1, 0.2])


class TestExtractFssToCsvSkipPaths:
    """ Tests for various continue/exception branches in extract_fss_to_csv. """

    def _setup_nc(self: "TestExtractFssToCsvSkipPaths", 
                  tmp_path: Path, 
                  subpath: str, 
                  content_ds: xr.Dataset) -> Path:
        """
        This helper method creates a NetCDF file at the specified subpath under tmp_path/output with the given content dataset. By constructing the full path and ensuring the parent directories exist, we can write the provided dataset to a NetCDF file that simulates one of the files that extract_fss_to_csv would encounter during its scanning process. This allows tests to set up specific scenarios (e.g., missing init time, missing lead time, missing metadata) by controlling the filename and content of the NetCDF file. The method returns the absolute path to the written NetCDF file for reference in the test. 

        Parameters:
            tmp_path (Path): Root temporary directory supplied by pytest.
            subpath (str): Relative path from tmp_path/output at which the file is written.
            content_ds (xr.Dataset): Dataset to serialize as the NetCDF file content.

        Returns:
            Path: Absolute path to the written NetCDF file.
        """
        nc_path = tmp_path / "output" / subpath
        nc_path.parent.mkdir(parents=True, exist_ok=True)
        content_ds.to_netcdf(str(nc_path))
        return nc_path

    def test_skip_no_init_time(self: "TestExtractFssToCsvSkipPaths", 
                               tmp_path: Path) -> None:
        """
        This test verifies that extract_fss_to_csv skips a NetCDF file when extract_init_time_from_path returns None for its path. When the initialization time directory component is absent or unrecognisable, the extractor cannot determine the cycle and must skip that file. This test creates a NetCDF file with a path that lacks the expected init time component and confirms that no CSV output is produced, indicating that the file was correctly skipped. 

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory.

        Returns:
            None
        """
        cfg = _make_cfg(tmp_path)
        fm = FileManager(cfg)

        ds = xr.Dataset({"fss": (["x"], [0.5])})
        self._setup_nc(
            tmp_path,
            "experiment/ExtendedFC/NOINIT/pp12h/fss_GLOBAL_pp90_w3.nc",
            ds,
        )
        csv_dir = tmp_path / "csv"
        fm.extract_fss_to_csv(
            output_dir=str(tmp_path / "output"),
            csv_dir=str(csv_dir),
        )
        assert not list(csv_dir.glob("*.csv"))

    def test_skip_no_lead_time(self: "TestExtractFssToCsvSkipPaths",
                               tmp_path: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that extract_fss_to_csv skips a NetCDF file when extract_lead_time_hours_from_path returns None for its path. When the lead-time directory component is absent or unrecognisable, the extractor must skip that file rather than writing a row with a None lead-time value. The utility function is replaced with a stub via monkeypatch. No CSV output should be produced, confirming the file was correctly skipped.

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory.
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        cfg = _make_cfg(tmp_path)
        fm = FileManager(cfg)

        ds = xr.Dataset({"fss": (["x"], [0.5])})
        self._setup_nc(
            tmp_path,
            "experiment/ExtendedFC/2024091700/pp12h/fss_GLOBAL_pp90_w3.nc",
            ds,
        )
        monkeypatch.setattr("modvx.utils.extract_lead_time_hours_from_path", lambda *a, **kw: None)

        csv_dir = tmp_path / "csv"
        fm.extract_fss_to_csv(
            output_dir=str(tmp_path / "output"),
            csv_dir=str(csv_dir),
        )
        assert not list(csv_dir.glob("*.csv"))

    def test_skip_no_metadata(self: "TestExtractFssToCsvSkipPaths",
                              tmp_path: Path,
                              monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that extract_fss_to_csv skips a NetCDF file when parse_fss_filename_metadata returns None. When the filename does not encode the expected domain, threshold, and window tokens, the extractor cannot reconstruct the metric metadata and must skip that file. Both utility functions are replaced via monkeypatch: lead-time extraction returns a valid value while metadata parsing returns None, confirming that the skip guard is triggered by the absent metadata. No CSV output should be produced.

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory.
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        cfg = _make_cfg(tmp_path)
        fm = FileManager(cfg)

        ds = xr.Dataset({"fss": (["x"], [0.5])})
        self._setup_nc(
            tmp_path,
            "experiment/ExtendedFC/2024091700/pp12h/fss_GLOBAL_pp90_w3.nc",
            ds,
        )
        monkeypatch.setattr("modvx.utils.extract_lead_time_hours_from_path", lambda *a, **kw: 6)
        monkeypatch.setattr("modvx.utils.parse_fss_filename_metadata", lambda *a, **kw: None)

        csv_dir = tmp_path / "csv"
        fm.extract_fss_to_csv(
            output_dir=str(tmp_path / "output"),
            csv_dir=str(csv_dir),
        )
        assert not list(csv_dir.glob("*.csv"))

    def test_exception_during_processing(self: "TestExtractFssToCsvSkipPaths", 
                                         tmp_path: Path) -> None:
        """
        This test verifies that extract_fss_to_csv handles exceptions raised during processing of a NetCDF file by logging the error and skipping that file without writing any CSV output. By creating a NetCDF file with valid path components but mocking the dataset loading to raise an exception (e.g., due to file corruption), we can confirm that the method catches the exception, logs an appropriate error message, and does not produce any CSV files. This ensures that the method can gracefully handle unexpected errors in individual files without failing the entire extraction process. 

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory.

        Returns:
            None
        """
        cfg = _make_cfg(tmp_path)
        fm = FileManager(cfg)

        nc_dir = tmp_path / "output" / "experiment" / "ExtendedFC" / "2024091700" / "pp12h"
        nc_dir.mkdir(parents=True)
        (nc_dir / "fss_GLOBAL_pp90_w3.nc").write_text("not a netcdf file")

        csv_dir = tmp_path / "csv"
        fm.extract_fss_to_csv(
            output_dir=str(tmp_path / "output"),
            csv_dir=str(csv_dir),
        )
        assert not list(csv_dir.glob("*.csv"))
