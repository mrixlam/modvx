#!/usr/bin/env python3

"""
Tests for the DataValidator class in the MODvx package. 

This module contains unit tests for the various methods of the DataValidator class, which is responsible for preparing forecast and observation data for verification. The tests cover coordinate standardization, mask extent computation, observation clipping, regridding to a common grid, and application of the verification mask. Each test verifies that the expected transformations are applied correctly to synthetic input data, ensuring that the DataValidator functions as intended before being used in the full verification pipeline. 

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
    This fixture provides a DataValidator instance configured with default settings for use in multiple test cases. By centralizing the creation of the DataValidator, we ensure that all tests operate with a consistent configuration and avoid redundant setup code. The default ModvxConfig can be customized in individual tests if needed by creating a new DataValidator instance with specific parameters. 

    Parameters:
        None

    Returns:
        DataValidator: A validator instance configured with default settings.
    """
    return DataValidator(ModvxConfig())


class TestPrepareObsCoords:
    """ Tests for prepare_observation_coordinates verifying that coordinate renaming and longitude wrapping are performed correctly. """

    def test_renames_lat_lon(self: "TestPrepareObsCoords") -> None:
        """
        This test verifies that prepare_observation_coordinates correctly renames latitude and longitude dimensions to 'latitude' and 'longitude'. It creates a DataArray with non-standard dimension names ('lat', 'lon') and checks that the output dimensions are renamed as expected. Proper coordinate naming is essential for downstream processing steps that rely on standard dimension names. 

        Parameters:
            None

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

    def test_negative_lon_converted(self: "TestPrepareObsCoords") -> None:
        """
        This test checks that prepare_observation_coordinates converts negative longitudes to the 0-360 range. It creates a DataArray with longitude coordinates that include negative values and asserts that the minimum longitude in the output is non-negative. Proper longitude wrapping is crucial for ensuring that observations align correctly with forecast grids, especially when verifying over global domains. 

        Parameters:
            None

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
    """ Tests for compute_mask_extent verifying correct bounding box extraction from a binary mask and error handling for empty masks. """

    def test_valid_extent(self: "TestMaskExtent") -> None:
        """
        This test verifies that compute_mask_extent correctly identifies the bounding box of non-zero cells in a binary mask. It creates a 2×3 mask with a specific pattern of zeros and ones, then asserts that the computed latitude and longitude extents match the expected values. Accurate bounding box extraction is necessary for efficient clipping of observation data to the relevant verification region, reducing computational load in subsequent steps. 

        Parameters:
            None

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

    def test_empty_mask_raises(self: "TestMaskExtent") -> None:
        """
        This test checks that compute_mask_extent raises a ValueError when the input mask contains no non-zero cells. It creates a 2×3 mask of all zeros and asserts that the expected exception is raised with an appropriate error message. Proper error handling for empty masks prevents downstream processing from proceeding with invalid spatial extents, which could lead to misleading verification results or runtime errors.

        Parameters:
            None

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
    """ Tests for clip_observations verifying that the output spatial extent includes the specified buffer padding beyond the mask-derived bounding box. """

    def test_clip_with_buffer(self: "TestClipObservations", 
                              dv: DataValidator) -> None:
        """
        This test verifies that clip_observations correctly clips the input observation to an extent that includes the specified buffer padding. It creates a 180×360 observation array with regular latitude and longitude coordinates, then calls the clipping method with a bounding box and buffer. The test asserts that the minimum and maximum latitude of the clipped output extend beyond the original bounding box by at least the buffer amount. Ensuring that the clipping includes the buffer is important for avoiding edge effects in verification when observations near the boundary of the mask are relevant.

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
    """ Tests for regrid_to_common_grid verifying that forecast and observation arrays are resampled to matching shapes based on the specified target resolution. """

    def test_obs_target(self: "TestRegrid") -> None:
        """
        This test verifies that regrid_to_common_grid with target_resolution='obs' resamples the forecast to the observation grid shape. It creates a coarse 5×5 forecast grid and a finer 3×3 observation grid, then asserts that the returned forecast array has the same shape as the observation array. Correct 'obs' target behavior is required when the pipeline is configured to verify on the observation grid rather than the native model grid. 

        Parameters:
            None

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

    def test_numeric_resolution(self: "TestRegrid") -> None:
        """
        This test verifies that regrid_to_common_grid with a numeric target_resolution resamples both forecast and observation to matching shapes on a common grid. It creates two 10×10 grids at 1°-resolution and requests resampling to a 2°-resolution common grid, then asserts that the two output arrays have equal shapes. Equal output shapes for numeric resolution targets are required before computing fraction fields over paired grid cells. 

        Parameters:
            None

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
    """ Tests for apply_verification_mask verifying that values outside the mask are set to NaN while values inside the mask are retained. """

    def test_nan_outside_mask(self: "TestApplyMask") -> None:
        """
        This test verifies that apply_verification_mask correctly applies NaN masking based on the provided binary mask. It creates a 3×3 forecast and observation field of ones, along with a mask that has ones on the diagonal and zeros elsewhere. The test asserts that the masked forecast has NaN values at the positions where the mask is zero and retains the original values where the mask is one. Proper mask application is essential for ensuring that only valid grid points are included in verification calculations. 

        Parameters:
            None

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


class TestRegridToTargetFcst:
    """ Tests for regrid_to_common_grid with target_resolution='fcst' verifying that the forecast grid shape is preserved after regridding to the observation grid. """

    def _make_grid(self: "TestRegridToTargetFcst",
                   lat_range: tuple = (0, 5),
                   lon_range: tuple = (0, 5),
                   step: float = 1.0,) -> xr.DataArray:
        """
        This helper method creates a synthetic 2-D DataArray with specified latitude and longitude ranges and grid spacing. The generated grid contains random values and is used in the regridding tests to simulate forecast and observation fields. By adjusting the lat_range, lon_range, and step parameters, we can create grids of varying resolutions and extents to thoroughly test the regridding functionality. 

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

    def test_regrid_to_target_fcst(self: "TestRegridToTargetFcst") -> None:
        """
        This test verifies that regrid_to_common_grid with target_resolution='fcst' preserves the forecast grid shape after regridding to the observation grid. It creates a 5×5 forecast grid and a finer 3×3 observation grid, then asserts that the returned forecast array has the same shape as the original forecast array. Preserving the forecast grid shape is important when the verification is configured to operate on the native model grid rather than resampling to the observation grid. 

        Parameters:
            None

        Returns:
            None
        """
        dv = DataValidator(ModvxConfig(target_resolution="obs"))
        fcst = self._make_grid(step=1.0)
        obs = self._make_grid(step=0.5)
        f_r, o_r = dv.regrid_to_common_grid(fcst, obs, "fcst", 0.0, 5.0, 0.0, 5.0)
        assert f_r.shape == fcst.shape

    def test_regrid_to_target_numeric(self: "TestRegridToTargetFcst") -> None:
        """
        This test verifies that regrid_to_common_grid with a numeric target_resolution resamples both forecast and observation to matching shapes on a common grid. It creates two 10×10 grids at 1°-resolution and requests resampling to a 2°-resolution common grid, then asserts that the two output arrays have equal shapes. Equal output shapes for numeric resolution targets are required before computing fraction fields over paired grid cells. 

        Parameters:
            None

        Returns:
            None
        """
        dv = DataValidator(ModvxConfig(target_resolution="obs"))
        fcst = self._make_grid(step=1.0)
        obs = self._make_grid(step=0.5)
        f_r, o_r = dv.regrid_to_common_grid(fcst, obs, 2.0, 0.0, 5.0, 0.0, 5.0)
        assert f_r.shape == o_r.shape


class TestApplyVerificationMaskNumeric:
    """ Tests for apply_verification_mask with numeric target_resolution verifying NaN masking behavior. """

    def test_apply_verification_mask_numeric(self: "TestApplyVerificationMaskNumeric") -> None:
        """
        This test verifies that apply_verification_mask correctly applies NaN masking based on the provided binary mask when using a numeric target_resolution. It creates a 6×6 forecast and observation field of ones, along with a mask that has ones in the first row and zeros elsewhere. The test asserts that the masked forecast has NaN values in the rows where the mask is zero and retains the original values in the row where the mask is one. Proper mask application is essential for ensuring that only valid grid points are included in verification calculations, especially when resampling to a common grid with a specified resolution. 

        Parameters:
            None

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
    """ Integration test for the full prepare() pipeline from raw forecast and observation arrays through to masked, regridded output. """

    def test_prepare_full_pipeline(self: "TestPreparePipeline") -> None:
        """
        This test verifies the full prepare() pipeline of the DataValidator class, which includes standardizing observation coordinates, computing mask extent, clipping observations to the buffered extent, regridding to a common grid based on the target resolution, and applying the verification mask. It creates synthetic forecast and observation arrays with known shapes and a simple binary mask, then asserts that the final prepared forecast and observation arrays have matching shapes. This end-to-end test ensures that all components of the preparation pipeline work together correctly to produce valid inputs for FSS computation.

        Parameters:
            None

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
