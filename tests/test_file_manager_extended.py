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
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from modvx.config import ModvxConfig
from modvx.file_manager import FileManager


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def tmp_cfg(tmp_path: Path) -> ModvxConfig:
    """
    Create a ModvxConfig using tmp_path as base_dir so all file I/O writes are ephemeral. The fixture sets experiment_name and forecast_step_hours to stable test values, giving tests a realistic configuration without relying on any real directory structure. All fixtures and tests that need a backed-up config object depend on this fixture.

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
    Construct a FileManager backed by the tmp_cfg fixture. This fixture provides a ready-to-use FileManager that writes all files under the pytest temporary directory, ensuring no test artifacts are left on the real filesystem. Tests needing both configuration and file manager access can use this fixture directly.

    Parameters:
        tmp_cfg (ModvxConfig): Temporary configuration fixture.

    Returns:
        FileManager: FileManager instance configured under the temporary directory.
    """
    return FileManager(tmp_cfg)


# -----------------------------------------------------------------------
# load_region_mask
# -----------------------------------------------------------------------

class TestLoadRegionMask:
    """Tests for load_region_mask: loading, coordinate normalisation, and error paths."""

    def test_loads_mask_from_netcdf(self, fm: FileManager, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that load_region_mask correctly loads a NetCDF mask file and renames coordinates. The loaded DataArray should have 'latitude' and/or 'longitude' in its dimensions regardless of the original coordinate names, confirming that the coordinate normalisation step runs. The returned variable name should match the key used in the source dataset.

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

    def test_missing_mask_raises(self, fm: FileManager) -> None:
        """
        Verify that load_region_mask raises FileNotFoundError when the mask file does not exist. Requesting a non-existent path should immediately raise with a message containing 'Mask file not found' rather than a lower-level OS error. This gives callers an actionable error message for configuration issues such as a wrong mask directory path.

        Returns:
            None
        """
        with pytest.raises(FileNotFoundError, match="Mask file not found"):
            fm.load_region_mask("/nonexistent/mask.nc")

    def test_empty_mask_variable_raises(self, fm: FileManager, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that load_region_mask raises ValueError when the mask file exists but contains no data variables. A NetCDF file containing only coordinate variables and no actual mask arrays is unexpected and should be rejected with a clear 'No mask variable' message rather than silently returning None or an empty DataArray.

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


# -----------------------------------------------------------------------
# Observation path & cache key
# -----------------------------------------------------------------------

class TestObservationPath:
    """Tests for get_observation_filepath vintage fallback."""

    def test_fallback_to_first_vintage(self, fm: FileManager) -> None:
        """
        Verify that get_observation_filepath returns a path using the first vintage preference when no file exists. When no file matching any vintage exists in the observation directory, the method should return the path it would construct for the first-preference vintage (FNL) rather than raising an error. This allows callers to attempt to load the file and handle the missing case themselves.

        Returns:
            None
        """
        path = fm.get_observation_filepath("20240917")
        assert "FNL" in path  # first in obs_vintage_preference

    def test_existing_vintage_found(self, fm: FileManager, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that get_observation_filepath returns the path for an existing later-preference vintage file. When a file matching a non-default vintage such as LTE is present in the observation directory, the method should prefer it over the default FNL vintage. This ensures the observation loading logic correctly identifies available data files over fixed default paths.

        Returns:
            None
        """
        obs_dir = Path(tmp_cfg.base_dir) / tmp_cfg.obs_dir
        obs_dir.mkdir(parents=True, exist_ok=True)
        # Create a file with LTE vintage
        lte_name = (
            "IMERG.A01H.VLD20240917.S20240917T000000."
            "E20240917T235959.LTE.V07B.SRCHHR.X3600Y1800.R0p1.FMT.nc"
        )
        (obs_dir / lte_name).write_text("stub")
        path = fm.get_observation_filepath("20240917")
        assert "LTE" in path


class TestObsCacheKey:
    """Tests for _obs_cache_key deterministic key generation."""

    def test_key_format(self, fm: FileManager) -> None:
        """
        Verify that _obs_cache_key returns a deterministic string encoding the valid time and accumulation duration. The key format combines the valid-time string and the configured forecast_step_hours to produce a unique cache identifier. This test confirms the specific format 'obs_accum_YYYYMMDDhh_Nh' expected by both the memory cache and disk cache file-naming logic.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 17, 6, 0)
        key = fm._observation_cache_key(vt)
        assert key == "obs_accum_2024091706_1h"


# -----------------------------------------------------------------------
# save_intermediate_precip
# -----------------------------------------------------------------------

class TestSaveIntermediatePrecip:
    """Tests for save_intermediate_precip writing debug NetCDF files."""

    def test_writes_netcdf(self, fm: FileManager, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that save_intermediate_precip creates a NetCDF file containing forecast, observation, and difference variables. The method should write the file to the configured debug directory using the cycle start and valid time for the path sub-structure. This test confirms the three expected variable names are present in the output dataset to ensure downstream diagnostic tools can read the file.

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


# -----------------------------------------------------------------------
# save_intermediate_binary
# -----------------------------------------------------------------------

class TestSaveIntermediateBinary:
    """Tests for save_intermediate_binary writing debug binary mask NetCDF files."""

    def test_writes_netcdf(self, fm: FileManager, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that save_intermediate_binary creates a NetCDF file containing forecast_binary and observation_binary variables. Binary mask intermediate files are written to the debug directory to support offline inspection of threshold-exceedance fields. This test confirms both expected variables are present in the output dataset at the expected cycle-start and valid-time path.

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


# -----------------------------------------------------------------------
# save_fss_results
# -----------------------------------------------------------------------

class TestSaveFssResults:
    """Tests for save_fss_results writing FSS NetCDF files."""

    def test_writes_fss_only(self, fm: FileManager, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that save_fss_results writes a NetCDF file containing only the FSS variable. The method should create one file per cycle-region-threshold-window combination and include a fss array of the correct length. This test confirms both that the file is created and that the expected variable is present with the correct number of lead-time entries.

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

    def test_filename_encoding(self, fm: FileManager, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that the output NetCDF filename encodes the region name, threshold value, and window size. The filename is parsed by extract_fss_to_csv to recover these parameters without opening the file, so the encoding must be consistent. This test uses a non-default region, a fractional threshold, and a non-default window to confirm all three fields appear correctly in the name.

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


# -----------------------------------------------------------------------
# save_contingency_results
# -----------------------------------------------------------------------

class TestSaveContingencyResults:
    """Tests for save_contingency_results writing contingency NetCDF files."""

    def test_writes_contingency_metrics(self, fm: FileManager, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that save_contingency_results writes a NetCDF file containing contingency metric variables. The method should create one file per cycle-region-threshold combination and include pod, far, csi, fbias, and ets arrays of the correct length.

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

    def test_filename_encoding(self, fm: FileManager, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that the contingency NetCDF filename encodes the region name and threshold value without a window size.

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


# -----------------------------------------------------------------------
# Forecast cache layers
# -----------------------------------------------------------------------

class TestForecastCaching:
    """Tests for forecast cache paths in accumulate_forecasts."""

    def test_memory_cache_hit(self, tmp_path: Path) -> None:
        """
        Verify that accumulate_forecasts returns the cached array directly when a memory cache entry exists. The in-memory cache avoids invoking the raw accumulation path on repeated access for the same valid time.

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

        with patch.object(fm, "_compute_forecast_accumulation") as mock_compute:
            result = fm.accumulate_forecasts(vt, "2024091700")

        mock_compute.assert_not_called()
        np.testing.assert_array_equal(result.values, cached.values)

    def test_disk_cache_hit(self, tmp_path: Path) -> None:
        """
        Verify that accumulate_forecasts loads from a disk cache entry when the memory cache is empty. The disk cache allows reuse across processes without recomputing accumulations.

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

        with patch.object(fm, "_compute_forecast_accumulation") as mock_compute:
            result = fm.accumulate_forecasts(vt, "2024091700")

        mock_compute.assert_not_called()
        np.testing.assert_array_equal(result.values, da.values)


# -----------------------------------------------------------------------
# extract_fss_to_csv
# -----------------------------------------------------------------------

class TestExtractFssToCsv:
    """Tests for extract_fss_to_csv scanning and writing CSV files."""

    def _create_output_tree(self, base: Path) -> None:
        """
        Create a realistic output directory tree with separate FSS and contingency NetCDF files for extract_fss_to_csv tests. The tree mirrors the directory structure produced by the live pipeline: output/exp/ExtendedFC/init/pp1h/. Files are written with two lead-time entries so that the CSV extractor can exercise its full parsing logic.

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

    def test_csv_written(self, tmp_path: Path) -> None:
        """
        Verify that extract_fss_to_csv scans the output directory and writes a CSV file with the correct columns. The test creates a standard NetCDF output tree using _create_output_tree and then runs the extractor, expecting one CSV file to be produced with both FSS and contingency-table metric columns plus the leadTime column.

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

    def test_no_files_logs_zero(self, tmp_path: Path) -> None:
        """
        Verify that extract_fss_to_csv handles an empty output directory without raising an exception. When no NetCDF files are found in the output tree, the method should exit cleanly without writing any CSV files. This guards against crashes during pipeline runs where some cycles failed to produce output.

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


# -----------------------------------------------------------------------
# accumulate_observations cache layers
# -----------------------------------------------------------------------

class TestObservationCaching:
    """Tests for the multi-level observation cache in accumulate_observations."""

    def test_memory_cache_hit(self, fm: FileManager) -> None:
        """
        Verify that accumulate_observations returns the cached array directly when a memory cache entry exists. The in-memory cache avoids reopening the observation file for each lead time within the same process. This test pre-populates the cache and confirms the returned values match without any file I/O.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 17, 6, 0)
        key = fm._observation_cache_key(vt)
        cached = xr.DataArray(np.ones(5))
        fm._obs_mem_cache[key] = cached

        result = fm.accumulate_observations(vt)
        np.testing.assert_array_equal(result.values, cached.values)

    def test_disk_cache_hit(self, fm: FileManager, tmp_cfg: ModvxConfig) -> None:
        """
        Verify that accumulate_observations loads from a disk-cached NetCDF file when the memory cache is empty. The disk cache persists observation accumulations across process restarts to avoid re-reading large IMERG files on subsequent runs. This test writes a fake cache file and confirms the loader returns the same data without calling the raw accumulation path.

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


# -----------------------------------------------------------------------
# Helper for tests that need full directory-based ModvxConfig
# -----------------------------------------------------------------------

def _make_cfg(tmp_path: Path) -> ModvxConfig:
    """
    Return a minimal ModvxConfig rooted in tmp_path with standard sub-directory names. This module-level helper is used by gap-closing tests that need a full configuration without access to the class-scoped fixtures. All directory keys are set to single-word relative paths to keep configuration construction concise.

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


# -----------------------------------------------------------------------
# accumulate_forecasts (delegates to mpas_reader)
# -----------------------------------------------------------------------


class TestAccumulateForecasts:
    """Cover FileManager.accumulate_forecasts which delegates to mpas_reader."""

    def test_accumulate_forecasts_calls_mpas_reader(self, tmp_path: Path) -> None:
        """
        Verify that accumulate_forecasts delegates to mpas_reader.load_mpas_precip and remap_to_latlon. The method should call load_mpas_precip twice (current and previous time step), subtract the cumulative precipitation values, and then call remap_to_latlon once to regrid to the regular lat-lon grid. This test mocks both reader functions and asserts the expected call counts and output shape.

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

        with patch("modvx.mpas_reader.load_mpas_precip") as mock_load, \
             patch("modvx.mpas_reader.remap_to_latlon") as mock_remap:
            mock_load.side_effect = [
                xr.DataArray(np.array([30.0, 50.0]), dims=["nCells"]),
                xr.DataArray(np.array([10.0, 20.0]), dims=["nCells"]),
            ]
            mock_remap.return_value = remapped

            vt = datetime.datetime(2024, 9, 17, 0)
            result = fm.accumulate_forecasts(vt, "2024091700")

            assert mock_load.call_count == 2
            mock_remap.assert_called_once()
            assert result.shape == (3, 3)


# -----------------------------------------------------------------------
# Disk cache write in accumulate_observations
# -----------------------------------------------------------------------


class TestObsDiskCacheWrite:
    """Cover the disk cache WRITE path in accumulate_observations."""

    def test_disk_cache_written(self, tmp_path: Path) -> None:
        """
        Verify that accumulate_observations writes a disk cache entry on the first call and uses the memory cache on the second. After the raw accumulator returns a DataArray, the result should be saved as a NetCDF file in the configured cache directory. A subsequent call for the same valid time should hit the in-memory cache and never invoke the raw accumulator again, confirming both cache layers work end-to-end.

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

        with patch.object(fm, "_compute_observation_accumulation_raw", return_value=fake_da):
            fm.accumulate_observations(vt)

        key = fm._observation_cache_key(vt)
        disk_path = os.path.join(cache_dir, f"{key}.nc")
        assert os.path.exists(disk_path)

        # Second call hits the memory cache
        with patch.object(fm, "_compute_observation_accumulation_raw") as mock_raw:
            fm.accumulate_observations(vt)
            mock_raw.assert_not_called()


# -----------------------------------------------------------------------
# _accumulate_observations_raw
# -----------------------------------------------------------------------


class TestAccumulateObservationsRaw:
    """Cover FileManager._compute_observation_accumulation_raw."""

    def test_raw_accumulation(self, tmp_path: Path) -> None:
        """
        Verify that _accumulate_observations_raw reads the observation file and returns a 2D spatial DataArray. The raw accumulator reads the observation NetCDF, sums over the forecast step hours, and returns a field with latitude and longitude dimensions. This test mocks the file path lookup and provides a synthetic dataset to confirm the shape and non-None return value without real IMERG files.

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

        with patch.object(fm, "get_observation_filepath") as mock_path:
            mock_path.return_value = str(tmp_path / "fake_obs.nc")
            ds.to_netcdf(str(tmp_path / "fake_obs.nc"))
            result = fm._compute_observation_accumulation_raw(vt)

        assert result is not None
        assert result.shape == (5, 5)


# -----------------------------------------------------------------------
# extract_fss_to_csv: legacy format and multi-metric format
# -----------------------------------------------------------------------


class TestExtractFssToCsvLegacy:
    """Cover the legacy single-array format branch in extract_fss_to_csv."""

    def test_legacy_format_extraction(self, tmp_path: Path) -> None:
        """
        Verify that extract_fss_to_csv correctly handles NetCDF files containing only a single FSS array without other metrics. The legacy format stores FSS as a 1D array along the lead_time dimension with no contingency-table columns; the extractor should read the FSS values and fill all other metric columns with NaN. This ensures backward compatibility with output from older pipeline versions.

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

    def test_multi_metric_format_extraction(self, tmp_path: Path) -> None:  # noqa: PLR0915
        """
        Verify that extract_fss_to_csv correctly reads a multi-metric NetCDF file and writes all six metric columns to the output CSV. The NetCDF dataset contains fss, pod, far, csi, fbias, and ets arrays for two lead times; the extractor must read all columns, fill the domain/threshold/ window metadata from the filename, and write a two-row CSV. This integration test confirms that the multi-metric format is handled end-to-end without any columns being silently omitted.

        Parameters:
            tmp_path (Path): Pytest-supplied per-test temporary directory.

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


# -----------------------------------------------------------------------
# Internal parsing helpers
# -----------------------------------------------------------------------

class TestFileManagerParsingHelpers:
    """Tests for internal parsing helper branches."""

    def test_extract_file_context_no_output(self) -> None:
        """
        Verify that _extract_file_context returns None when the path lacks an 'output' segment.

        Returns:
            None
        """
        result = FileManager._extract_file_context("/tmp/experiment/ExtendedFC/2024091700/pp1h/file.nc")
        assert result is None

    def test_parse_metric_values_fallback_length(self) -> None:
        """
        Verify that _parse_metric_values falls back to the first data variable for length when no metric key matches.

        Returns:
            None
        """
        ds = xr.Dataset({"other": xr.DataArray([1.0, 2.0, 3.0], dims=["x"])})
        length, metrics = FileManager._parse_metric_values(ds, ["fss"])
        assert length == 3
        assert len(metrics["fss"]) == 3
        assert np.isnan(metrics["fss"]).all()

    def test_parse_metric_values_legacy_fss(self) -> None:
        """
        Verify that _parse_metric_values reads legacy FSS values from __xarray_dataarray_variable__.

        Returns:
            None
        """
        ds = xr.Dataset({
            "__xarray_dataarray_variable__": xr.DataArray([0.1, 0.2], dims=["x"])
        })
        length, metrics = FileManager._parse_metric_values(ds, ["fss"])
        assert length == 2
        np.testing.assert_allclose(metrics["fss"], [0.1, 0.2])


# -----------------------------------------------------------------------
# extract_fss_to_csv skip/exception branches
# -----------------------------------------------------------------------


class TestExtractFssToCsvSkipPaths:
    """Cover various continue/exception branches in extract_fss_to_csv."""

    def _setup_nc(
        self, tmp_path: Path, subpath: str, content_ds: xr.Dataset
    ) -> Path:
        """
        Write a NetCDF dataset to a path under tmp_path/output and return the full path. This helper creates any required intermediate directories and delegates to xarray's to_netcdf for writing. It is shared by all skip-branch tests in this class to avoid repeating directory creation and file-writing boilerplate.

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

    def test_skip_no_init_time(self, tmp_path: Path) -> None:
        """
        Verify that extract_fss_to_csv skips NetCDF files whose parent path contains no recognisable 8-digit initialisation-time directory component. The directory labelled 'NOINIT' does not match the expected YYYYMMDDhh pattern, so the extractor should skip the file entirely and write no CSV output. This guards against pipeline runs where an unexpected directory structure could otherwise cause malformed rows with a None init time to be written to the CSV.

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

    def test_skip_no_lead_time(self, tmp_path: Path) -> None:
        """
        Verify that extract_fss_to_csv skips a NetCDF file when extract_lead_time_hours returns None for its path. When the lead-time directory component is absent or unrecognisable, the extractor must skip that file rather than writing a row with a None lead-time value. This test mocks the lead-time extractor to return None and confirms no CSV output is produced.

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
            "experiment/ExtendedFC/2024091700/pp12h/fss_GLOBAL_pp90_w3.nc",
            ds,
        )
        with patch("modvx.utils.extract_lead_time_hours_from_path", return_value=None):
            csv_dir = tmp_path / "csv"
            fm.extract_fss_to_csv(
                output_dir=str(tmp_path / "output"),
                csv_dir=str(csv_dir),
            )
        assert not list(csv_dir.glob("*.csv"))

    def test_skip_no_metadata(self, tmp_path: Path) -> None:
        """
        Verify that extract_fss_to_csv skips a NetCDF file when parse_filename_metadata returns None. When the filename does not encode the expected domain, threshold, and window tokens, the extractor cannot reconstruct the metric metadata and must skip that file. This test mocks both lead-time extraction (to return a valid value) and filename metadata parsing (to return None) to confirm the skip logic is triggered by the metadata guard.

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
            "experiment/ExtendedFC/2024091700/pp12h/fss_GLOBAL_pp90_w3.nc",
            ds,
        )
        with patch(
            "modvx.utils.extract_lead_time_hours_from_path",
            return_value=6,
        ), patch(
            "modvx.utils.parse_fss_filename_metadata",
            return_value=None,
        ):
            csv_dir = tmp_path / "csv"
            fm.extract_fss_to_csv(
                output_dir=str(tmp_path / "output"),
                csv_dir=str(csv_dir),
            )
        assert not list(csv_dir.glob("*.csv"))

    def test_exception_during_processing(self, tmp_path: Path) -> None:
        """
        Verify that extract_fss_to_csv handles a corrupt or unreadable NetCDF file without raising  an exception. When xarray fails to open a file due to invalid content, the extractor should catch the exception, skip that file, and continue processing any remaining entries. This test writes a plain text file with a .nc extension to simulate a corrupt NetCDF and confirms no CSV is produced and no exception propagates to the caller.

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
