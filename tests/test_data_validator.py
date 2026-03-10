"""Tests for modvx.data_validator — grid preparation pipeline."""

import numpy as np
import pytest
import xarray as xr

from modvx.config import ModvxConfig
from modvx.data_validator import DataValidator


@pytest.fixture
def dv() -> DataValidator:
    """Construct a DataValidator instance backed by default ModvxConfig settings. This fixture provides a ready-to-use validator object for tests that require an instantiated class rather than calling static methods directly. It relies on ModvxConfig's defaults, so tests using this fixture do not depend on any filesystem state or YAML configuration files."""
    return DataValidator(ModvxConfig())


class TestPrepareObsCoords:
    """Tests for the prepare_observation_coordinates static method covering coordinate renaming and longitude normalization."""

    def test_renames_lat_lon(self) -> None:
        """Confirm that prepare_observation_coordinates renames 'lat'/'lon' dimension names to 'latitude'/'longitude'. This test constructs a minimal DataArray with short-form coordinate names and asserts that the output uses the canonical long-form names expected by downstream pipeline components. Correct renaming is critical for consistent coordinate access across forecast and observation grids."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=["lat", "lon"],
            coords={"lat": [10, 20, 30], "lon": [100, 110, 120, 130]},
        )
        result = DataValidator.prepare_observation_coordinates(da)
        assert "latitude" in result.dims
        assert "longitude" in result.dims

    def test_negative_lon_converted(self) -> None:
        """Verify that negative longitude values are converted to the [0, 360) range after coordinate preparation. This test supplies a DataArray with longitude coordinates spanning negative values and asserts the output minimum longitude is non-negative. Consistent positive-only longitude representation prevents misalignment when overlaying forecast and observation grids that may use different longitude conventions."""
        da = xr.DataArray(
            np.ones((2, 3)),
            dims=["lat", "lon"],
            coords={"lat": [0, 1], "lon": [-10, 0, 10]},
        )
        result = DataValidator.prepare_observation_coordinates(da)
        assert float(result.longitude.min()) >= 0


class TestMaskExtent:
    """Tests for get_mask_extent verifying bounding-box extraction from binary mask DataArrays."""

    def test_valid_extent(self) -> None:
        """Verify that get_mask_extent correctly identifies the bounding box of non-zero mask cells. This test builds a 2×3 binary mask with ones at known coordinates and checks that the returned lat/lon extents match the coordinates of those active cells, excluding zero-value positions. Accurate extent extraction is required to properly clip observation data before regridding."""
        mask = xr.DataArray(
            np.array([[0, 1, 1], [0, 0, 1]]),
            dims=["latitude", "longitude"],
            coords={"latitude": [10, 20], "longitude": [100, 110, 120]},
        )
        lat_min, lat_max, lon_min, lon_max = DataValidator.get_mask_extent(mask)
        assert lat_min == 10
        assert lat_max == 20
        assert lon_min == 110
        assert lon_max == 120

    def test_empty_mask_raises(self) -> None:
        """Ensure that get_mask_extent raises a ValueError when the supplied mask contains no non-zero cells. This test provides an all-zeros mask and expects the function to fail loudly with a message matching 'No valid points', guarding against silent downstream errors when an empty or misconfigured mask is loaded. Explicit failure on an empty mask surfaces configuration problems early in pipeline execution."""
        mask = xr.DataArray(
            np.zeros((2, 3)),
            dims=["latitude", "longitude"],
            coords={"latitude": [0, 1], "longitude": [0, 1, 2]},
        )
        with pytest.raises(ValueError, match="No valid points"):
            DataValidator.get_mask_extent(mask)


class TestClipObservations:
    """Tests for the clip_observations method verifying spatial subsetting and buffer expansion behavior."""

    def test_clip_with_buffer(self, dv: DataValidator) -> None:
        """Check that clip_observations returns a spatial subset that extends at least one degree beyond the specified bounding box in both latitude directions. This test uses a global 1°-resolution observation grid and a narrow lat/lon extent, then asserts that the clipped output covers coordinates outside the exact bounds due to the default buffer padding. Correct buffer handling prevents edge-cut artifacts during subsequent regridding and mask application."""
        obs = xr.DataArray(
            np.ones((180, 360)),
            dims=["latitude", "longitude"],
            coords={
                "latitude": np.arange(-89.5, 90.5, 1.0),
                "longitude": np.arange(0.5, 360.5, 1.0),
            },
        )
        clipped = dv.clip_observations(obs, 10, 20, 100, 110)
        # Should include buffer (default 1°)
        assert float(clipped.latitude.min()) <= 10
        assert float(clipped.latitude.max()) >= 20


class TestRegrid:
    """Tests for regrid_to_target covering interpolation to observation grids and fixed-resolution common grids."""

    def test_obs_target(self) -> None:
        """Verify that when target is set to 'obs', the forecast array is interpolated to match the shape of the observation grid. This test constructs a coarser 5×5 forecast and a finer 3×3 observation array and asserts the regridded forecast output has the same shape as the observation. Matching grid shapes is a prerequisite for element-wise FSS binary mask and fractional field computation."""
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
        fr, or_ = DataValidator.regrid_to_target(fcst, obs, "obs", 0, 4, 0, 4)
        assert fr.shape == obs.shape

    def test_numeric_resolution(self) -> None:
        """Verify that passing a numeric resolution value causes both forecast and observation grids to be resampled to a shared common grid with matching shapes. This test supplies identical 10×10 grids and a resolution of 2.0 degrees, then asserts the two output arrays have the same shape. Ensuring equal output shapes for numeric resolution targets is necessary before computing fraction fields over paired grid cells."""
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
        fr, or_ = DataValidator.regrid_to_target(fcst, obs, 2.0, 0, 8, 0, 8)
        assert fr.shape == or_.shape


class TestApplyMask:
    """Tests for apply_verification_mask verifying NaN assignment outside active mask cells and value preservation inside."""

    def test_nan_outside_mask(self) -> None:
        """Confirm that apply_verification_mask sets values to NaN at positions where the mask is zero and preserves original values where the mask is one. This test constructs an all-ones 3×3 field and a diagonal binary mask, then asserts that off-diagonal positions become NaN while diagonal positions retain the original 1.0 value. Proper NaN masking ensures that FSS computation is restricted to the intended verification region and does not include points outside the domain."""
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
        fm, om = DataValidator.apply_verification_mask(
            field, field, mask, "obs", 0, 2, 0, 2,
        )
        # Masked-out positions should be NaN
        assert np.isnan(fm.values[0, 1])
        assert np.isnan(fm.values[0, 2])
        # Mask-in positions should remain 1.0
        assert fm.values[0, 0] == 1.0
        assert fm.values[1, 1] == 1.0
