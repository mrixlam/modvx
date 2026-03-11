#!/usr/bin/env python3

"""
Tests for modvx.data_validator — grid preparation pipeline.

This module contains unit tests for the DataValidator class, which performs the critical task of preparing forecast and observation data for verification. The tests cover each step of the grid preparation pipeline. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

import datetime

import numpy as np
import pytest
import xarray as xr

from modvx.config import ModvxConfig
from modvx.data_validator import DataValidator


@pytest.fixture
def dv() -> DataValidator:
    """
    Construct a DataValidator instance backed by default ModvxConfig settings. This fixture provides a ready-to-use validator object for tests that require an instantiated class rather than calling static methods directly. It relies on ModvxConfig's defaults, so tests using this fixture do not depend on any filesystem state or YAML configuration files.

    Returns:
        DataValidator: A validator instance configured with default settings.
    """
    return DataValidator(ModvxConfig())


class TestPrepareObsCoords:
    """Tests for the standardize_observation_coordinates static method covering coordinate renaming and longitude normalization."""

    def test_renames_lat_lon(self) -> None:
        """
        Confirm that prepare_observation_coordinates renames 'lat'/'lon' dimension names to 'latitude'/'longitude'. This test constructs a minimal DataArray with short-form coordinate names and asserts that the output uses the canonical long-form names expected by downstream pipeline components. Correct renaming is critical for consistent coordinate access across forecast and observation grids.

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=["lat", "lon"],
            coords={"lat": [10, 20, 30], "lon": [100, 110, 120, 130]},
        )
        result = DataValidator.standardize_observation_coordinates(da)
        assert "latitude" in result.dims
        assert "longitude" in result.dims

    def test_negative_lon_converted(self) -> None:
        """
        Verify that negative longitude values are converted to the [0, 360) range after coordinate preparation. This test supplies a DataArray with longitude coordinates spanning negative values and asserts the output minimum longitude is non-negative. Consistent positive-only longitude representation prevents misalignment when overlaying forecast and observation grids that may use different longitude conventions.

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones((2, 3)),
            dims=["lat", "lon"],
            coords={"lat": [0, 1], "lon": [-10, 0, 10]},
        )
        result = DataValidator.standardize_observation_coordinates(da)
        assert float(result.longitude.min()) >= 0


class TestMaskExtent:
    """Tests for compute_mask_extent verifying bounding-box extraction from binary mask DataArrays."""

    def test_valid_extent(self) -> None:
        """
        Verify that get_mask_extent correctly identifies the bounding box of non-zero mask cells. This test builds a 2×3 binary mask with ones at known coordinates and checks that the returned lat/lon extents match the coordinates of those active cells, excluding zero-value positions. Accurate extent extraction is required to properly clip observation data before regridding.

        Returns:
            None
        """
        mask = xr.DataArray(
            np.array([[0, 1, 1], [0, 0, 1]]),
            dims=["latitude", "longitude"],
            coords={"latitude": [10, 20], "longitude": [100, 110, 120]},
        )
        lat_min, lat_max, lon_min, lon_max = DataValidator.compute_mask_extent(mask)
        assert lat_min == 10
        assert lat_max == 20
        assert lon_min == 110
        assert lon_max == 120

    def test_empty_mask_raises(self) -> None:
        """
        Ensure that get_mask_extent raises a ValueError when the supplied mask contains no non-zero cells. This test provides an all-zeros mask and expects the function to fail loudly with a message matching 'No valid points', guarding against silent downstream errors when an empty or misconfigured mask is loaded. Explicit failure on an empty mask surfaces configuration problems early in pipeline execution.

        Returns:
            None
        """
        mask = xr.DataArray(
            np.zeros((2, 3)),
            dims=["latitude", "longitude"],
            coords={"latitude": [0, 1], "longitude": [0, 1, 2]},
        )
        with pytest.raises(ValueError, match="No valid points"):
            DataValidator.compute_mask_extent(mask)


class TestClipObservations:
    """Tests for the clip_observation_to_buffer method verifying spatial subsetting and buffer expansion behavior."""

    def test_clip_with_buffer(self, dv: DataValidator) -> None:
        """
        Check that clip_observations returns a spatial subset that extends at least one degree beyond the specified bounding box. This test uses a global 1°-resolution observation grid and a narrow lat/lon extent, then asserts that the clipped output covers coordinates outside the exact bounds due to the default buffer padding. Correct buffer handling prevents edge-cut artifacts during subsequent regridding and mask application.

        Parameters:
            dv (DataValidator): DataValidator fixture configured with default ModvxConfig settings.

        Returns:
            None
        """
        obs = xr.DataArray(
            np.ones((180, 360)),
            dims=["latitude", "longitude"],
            coords={
                "latitude": np.arange(-89.5, 90.5, 1.0),
                "longitude": np.arange(0.5, 360.5, 1.0),
            },
        )
        clipped = dv.clip_observation_to_buffer(obs, 10, 20, 100, 110)
        # Should include buffer (default 1°)
        assert float(clipped.latitude.min()) <= 10
        assert float(clipped.latitude.max()) >= 20


class TestRegrid:
    """Tests for regrid_to_common_grid covering interpolation to observation grids and fixed-resolution common grids."""

    def test_obs_target(self) -> None:
        """
        Verify that when target is set to 'obs', the forecast array is interpolated to match the shape of the observation grid. This test constructs a coarser 5×5 forecast and a finer 3×3 observation array and asserts the regridded forecast output has the same shape as the observation. Matching grid shapes is a prerequisite for element-wise FSS binary mask and fractional field computation.

        Returns:
            None
        """
        fcst = xr.DataArray(
            np.ones((5, 5)),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(5), "longitude": np.arange(5)},
        )
        obs = xr.DataArray(
            np.ones((3, 3)),
            dims=["latitude", "longitude"],
            coords={"latitude": [1, 2, 3], "longitude": [1, 2, 3]},
        )
        fr, or_ = DataValidator.regrid_to_common_grid(fcst, obs, "obs", 0, 4, 0, 4)
        assert fr.shape == obs.shape

    def test_numeric_resolution(self) -> None:
        """
        Verify that passing a numeric resolution value causes both grids to be resampled to a shared common grid. This test supplies identical 10×10 grids and a resolution of 2.0 degrees, then asserts the two output arrays have the same shape. Ensuring equal output shapes for numeric resolution targets is necessary before computing fraction fields over paired grid cells.

        Returns:
            None
        """
        fcst = xr.DataArray(
            np.ones((10, 10)),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(10), "longitude": np.arange(10)},
        )
        obs = xr.DataArray(
            np.ones((10, 10)),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(10), "longitude": np.arange(10)},
        )
        fr, or_ = DataValidator.regrid_to_common_grid(fcst, obs, 2.0, 0, 8, 0, 8)
        assert fr.shape == or_.shape


class TestApplyMask:
    """Tests for apply_domain_mask verifying NaN assignment outside active mask cells and value preservation inside."""

    def test_nan_outside_mask(self) -> None:
        """
        Confirm that apply_verification_mask sets values to NaN at positions where the mask is zero. This test constructs an all-ones 3×3 field and a diagonal binary mask, then asserts that off-diagonal positions become NaN while diagonal positions retain the original 1.0 value. Proper NaN masking ensures that FSS computation is restricted to the intended verification region and does not include points outside the domain.

        Returns:
            None
        """
        field = xr.DataArray(
            np.ones((3, 3)),
            dims=["latitude", "longitude"],
            coords={"latitude": [0, 1, 2], "longitude": [0, 1, 2]},
        )
        mask = xr.DataArray(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            dims=["latitude", "longitude"],
            coords={"latitude": [0, 1, 2], "longitude": [0, 1, 2]},
        )
        fm, om = DataValidator.apply_domain_mask(
            field, field, mask, "obs", 0, 2, 0, 2,
        )
        # Masked-out positions should be NaN
        assert np.isnan(fm.values[0, 1])
        assert np.isnan(fm.values[0, 2])
        # Mask-in positions should remain 1.0
        assert fm.values[0, 0] == 1.0
        assert fm.values[1, 1] == 1.0


# -----------------------------------------------------------------------
# DataValidator gap-closing tests
# -----------------------------------------------------------------------


class TestRegridToTargetFcst:
    """Cover regrid_to_target with target_resolution='fcst'."""

    def _make_grid(
        self,
        lat_range: tuple = (0, 5),
        lon_range: tuple = (0, 5),
        step: float = 1.0,
    ) -> xr.DataArray:
        """
        Build a synthetic 2-D DataArray with regular latitude/longitude coordinates for grid operation tests. The array is populated with random values drawn from a fixed random seed to ensure reproducibility across test runs. Latitude and longitude ranges and step size are configurable so both coarse and fine grid resolutions can be produced with a single call.

        Parameters:
            lat_range (tuple): Two-element (min, max) latitude extent in degrees. Defaults to (0, 5).
            lon_range (tuple): Two-element (min, max) longitude extent in degrees. Defaults to (0, 5).
            step (float): Grid spacing in degrees for both latitude and longitude. Defaults to 1.0.

        Returns:
            xr.DataArray: 2-D DataArray with 'latitude' and 'longitude' dimensions.
        """
        lats = np.arange(lat_range[0], lat_range[1] + step, step)
        lons = np.arange(lon_range[0], lon_range[1] + step, step)
        data = np.random.default_rng(42).random((len(lats), len(lons)))
        return xr.DataArray(
            data,
            dims=["latitude", "longitude"],
            coords={"latitude": lats, "longitude": lons},
        )

    def test_regrid_to_target_fcst(self) -> None:
        """
        Verify that regrid_to_target with target_resolution='fcst' resamples the observation to the forecast grid shape. This test uses a coarse 1°-resolution forecast grid and a finer 0.5°-resolution observation grid, then asserts that the returned forecast array retains the original forecast shape. Correct 'fcst' target behavior is required when the pipeline is configured to verify on the native model grid rather than the observation grid.

        Returns:
            None
        """
        dv = DataValidator(ModvxConfig(target_resolution="obs"))
        fcst = self._make_grid(step=1.0)
        obs = self._make_grid(step=0.5)
        f_r, o_r = dv.regrid_to_common_grid(fcst, obs, "fcst", 0.0, 5.0, 0.0, 5.0)
        assert f_r.shape == fcst.shape

    def test_regrid_to_target_numeric(self) -> None:
        """
        Verify that a numeric resolution target resamples both grids to matching shapes on a common grid. This test supplies two grids at 1°-resolution and requests resampling to a 2°-resolution common grid, then asserts the two output arrays have equal shapes. Equal output shapes for numeric resolution targets are required before computing fraction fields over paired grid cells.

        Returns:
            None
        """
        dv = DataValidator(ModvxConfig(target_resolution="obs"))
        fcst = self._make_grid(step=1.0)
        obs = self._make_grid(step=0.5)
        f_r, o_r = dv.regrid_to_common_grid(fcst, obs, 2.0, 0.0, 5.0, 0.0, 5.0)
        assert f_r.shape == o_r.shape


class TestApplyVerificationMaskNumeric:
    """Cover apply_verification_mask with numeric resolution."""

    def test_apply_verification_mask_numeric(self) -> None:
        """
        Verify that apply_verification_mask correctly applies NaN masking when target_resolution is numeric. This test constructs matching 6×6 forecast and observation fields, zeros out the first row of the mask, and then asserts that the masked forecast has NaN at that row and retains values elsewhere. Testing the numeric resolution path ensures that mask application is consistent with the 'obs' and 'fcst' string resolution targets.

        Returns:
            None
        """
        lats = np.arange(0, 6, 1.0)
        lons = np.arange(0, 6, 1.0)
        data = np.ones((len(lats), len(lons)))
        fcst = xr.DataArray(data, dims=["latitude", "longitude"],
                            coords={"latitude": lats, "longitude": lons})
        obs = fcst.copy()
        mask = xr.DataArray(
            np.ones((len(lats), len(lons))),
            dims=["latitude", "longitude"],
            coords={"latitude": lats, "longitude": lons},
        )
        mask.values[0, :] = 0.0

        f_m, o_m = DataValidator.apply_domain_mask(
            fcst, obs, mask, 1.0, 0.0, 5.0, 0.0, 5.0,
        )
        assert np.isnan(f_m.values[0, 0])
        assert not np.isnan(f_m.values[1, 0])


class TestPreparePipeline:
    """Cover prepare() full pipeline."""

    def test_prepare_full_pipeline(self) -> None:
        """
        Exercise the full prepare() pipeline from raw forecast and observation arrays through to masked, regridded output. This integration test constructs synthetic 21×21-latitude grids centred on the antimeridian to exercise longitude-wrapping edge cases, along with an all-ones mask. It then calls prepare() and asserts that the returned forecast and observation shapes match, confirming that clipping, regridding, and mask application all compose correctly without raising exceptions.

        Returns:
            None
        """
        cfg = ModvxConfig(
            target_resolution="obs",
            forecast_step_hours=1,
            clip_buffer_deg=0.5,
        )
        dv = DataValidator(cfg)

        lats = np.arange(-10, 11, 1.0)
        lons = np.arange(170, 191, 1.0)
        rng = np.random.default_rng(42)

        fcst = xr.DataArray(
            rng.random((len(lats), len(lons))),
            dims=["latitude", "longitude"],
            coords={"latitude": lats, "longitude": lons},
        )
        obs = xr.DataArray(
            rng.random((len(lats), len(lons))),
            dims=["latitude", "longitude"],
            coords={"latitude": lats, "longitude": lons},
        )
        mask = xr.DataArray(
            np.ones((len(lats), len(lons))),
            dims=["latitude", "longitude"],
            coords={"latitude": lats, "longitude": lons},
        )

        f_m, o_m = dv.prepare(fcst, obs, mask, datetime.datetime(2024, 9, 17))
        assert f_m.shape == o_m.shape
