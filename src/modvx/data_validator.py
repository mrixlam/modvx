"""
Grid preparation and data validation for modvx.

``DataValidator`` handles every step between raw loaded data and the
analysis-ready, co-located, masked forecast/observation arrays that
``PerfMetrics`` expects:

1. Coordinate standardisation & longitude normalisation.
2. Mask-extent calculation.
3. Observation clipping (with configurable buffer).
4. Regridding to a common target grid.
5. Verification-mask application.
6. Shape / grid-match assertions.
"""

from __future__ import annotations

import datetime
import logging
from typing import Tuple, Union

import numpy as np
import xarray as xr

from .config import ModvxConfig
from .utils import normalize_longitude, standardize_coords

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Prepare forecast and observation fields for co-located, masked verification arrays.
    This class encapsulates every preprocessing step between raw loaded data and the
    analysis-ready arrays expected by PerfMetrics. Processing steps include coordinate
    standardisation, longitude normalisation, mask-extent calculation, observation
    clipping, regridding to a common grid, and verification-mask application.
    All behaviour is governed by the provided ModvxConfig instance.

    Parameters:
        config (ModvxConfig): Run configuration with target resolution, clip buffer size,
            and verbosity settings.
    """

    def __init__(self, config: ModvxConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_observation_coordinates(obs: xr.DataArray) -> xr.DataArray:
        """
        Standardise the coordinate names and longitude convention of an observation DataArray.
        Input observation files may use abbreviated dimension names (``lat``/``lon``) and arbitrary
        longitude conventions. This method applies standardize_coords to rename dimensions to
        ``latitude``/``longitude`` and then converts longitudes to the [0, 360] convention required
        by the clipping and regridding steps that follow. It is always the first step applied to
        raw observation data in the processing pipeline.

        Parameters:
            obs (xr.DataArray): Raw observation DataArray with potentially non-standard
                coordinate names.

        Returns:
            xr.DataArray: Observation DataArray with standard coordinate names and
                [0, 360] longitudes.
        """
        obs = standardize_coords(obs)
        obs = normalize_longitude(obs, "0_360")
        return obs

    # ------------------------------------------------------------------
    # Mask extent
    # ------------------------------------------------------------------

    @staticmethod
    def get_mask_extent(
        region_mask: xr.DataArray,
    ) -> Tuple[float, float, float, float]:
        """
        Compute the geographic bounding box of active (non-zero) points in a region mask.
        The bounding box is determined by finding the latitude and longitude coordinates that
        contain at least one active mask cell, then taking the min and max of each axis.
        The result is used to clip the observation field to the region of interest before
        regridding, reducing unnecessary computation. A ValueError is raised when the mask
        contains no active points, indicating a misconfigured or incompatible mask file.

        Parameters:
            region_mask (xr.DataArray): Binary verification-domain mask with
                ``latitude``/``longitude`` coordinates.

        Returns:
            Tuple[float, float, float, float]: Bounding box as
                ``(lat_min, lat_max, lon_min, lon_max)`` in degrees.

        Raises:
            ValueError: If the mask contains no active (non-zero) grid points.
        """
        mask_bool = region_mask.values > 0
        valid_lats = region_mask.latitude.values[mask_bool.any(axis=1)]
        valid_lons = region_mask.longitude.values[mask_bool.any(axis=0)]

        if len(valid_lats) == 0 or len(valid_lons) == 0:
            raise ValueError("No valid points in verification domain mask")

        lat_min, lat_max = float(valid_lats.min()), float(valid_lats.max())
        lon_min, lon_max = float(valid_lons.min()), float(valid_lons.max())

        logger.debug(
            "Mask extent: lat [%.2f, %.2f], lon [%.2f, %.2f]",
            lat_min, lat_max, lon_min, lon_max,
        )
        return lat_min, lat_max, lon_min, lon_max

    # ------------------------------------------------------------------
    # Clipping
    # ------------------------------------------------------------------

    def clip_observations(
        self,
        obs: xr.DataArray,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> xr.DataArray:
        """
        Clip an observation DataArray to the verification domain extent with an added buffer.
        A configurable buffer (default 1°, set via ModvxConfig.clip_buffer_deg) is added on all
        sides of the bounding box before slicing, ensuring that the subsequent regridding step
        has sufficient data near domain edges to produce accurate interpolated values. Latitude
        and longitude bounds are clamped to valid geographic ranges ([-90, 90] and [0, 360]).
        This significantly reduces data volume for regional domain verification.

        Parameters:
            obs (xr.DataArray): Global or large-domain observation DataArray with standard
                coordinate names.
            lat_min (float): Southern latitude boundary of the verification domain in degrees.
            lat_max (float): Northern latitude boundary of the verification domain in degrees.
            lon_min (float): Western longitude boundary in degrees (in [0, 360] convention).
            lon_max (float): Eastern longitude boundary in degrees (in [0, 360] convention).

        Returns:
            xr.DataArray: Observation DataArray clipped to the buffered domain extent.
        """
        buf = self.config.clip_buffer_deg
        clipped = obs.sel(
            latitude=slice(max(lat_min - buf, -90), min(lat_max + buf, 90)),
            longitude=slice(max(lon_min - buf, 0), min(lon_max + buf, 360)),
        )
        logger.debug(
            "Observation clipped: %d × %d",
            clipped.latitude.size,
            clipped.longitude.size,
        )
        return clipped

    # ------------------------------------------------------------------
    # Regridding
    # ------------------------------------------------------------------

    @staticmethod
    def regrid_to_target(
        fcst: xr.DataArray,
        obs_clipped: xr.DataArray,
        target_resolution: Union[str, float],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Regrid forecast and observation arrays to a common regular lat-lon grid.
        Three target resolution modes are supported: ``"obs"`` interpolates the forecast
        to match the observation grid; ``"fcst"`` interpolates the observation to the
        forecast grid; and a floating-point value creates a new regular grid at that
        degree spacing covering the domain bounds. Linear interpolation is used for
        floating-point targets. The returned arrays share identical coordinates.

        Parameters:
            fcst (xr.DataArray): Forecast precipitation field on its native grid.
            obs_clipped (xr.DataArray): Clipped observation DataArray on its native grid.
            target_resolution (str or float): Regrid target: ``"obs"``, ``"fcst"``,
                or a float degree spacing.
            lat_min (float): Southern latitude boundary of the target domain in degrees.
            lat_max (float): Northern latitude boundary of the target domain in degrees.
            lon_min (float): Western longitude boundary of the target domain in degrees.
            lon_max (float): Eastern longitude boundary of the target domain in degrees.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: Tuple of
                ``(fcst_regridded, obs_regridded)`` on the common grid.
        """
        if target_resolution == "obs":
            return fcst.interp_like(obs_clipped), obs_clipped
        if target_resolution == "fcst":
            return fcst, obs_clipped.interp_like(fcst)

        res = float(target_resolution)
        target_lats = np.arange(lat_min, lat_max + res, res)
        target_lons = np.arange(lon_min, lon_max + res, res)
        fcst_r = fcst.interp(latitude=target_lats, longitude=target_lons, method="linear")
        obs_r = obs_clipped.interp(latitude=target_lats, longitude=target_lons, method="linear")
        return fcst_r, obs_r

    # ------------------------------------------------------------------
    # Mask application
    # ------------------------------------------------------------------

    @staticmethod
    def apply_verification_mask(
        fcst: xr.DataArray,
        obs: xr.DataArray,
        region_mask: xr.DataArray,
        target_resolution: Union[str, float],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Apply a binary verification-domain mask to forecast and observation arrays.
        Grid points outside the mask (where mask value ≤ 0.5 after interpolation) are set
        to NaN in both arrays. The mask is first resampled to the data grid via
        nearest-neighbour interpolation so that its resolution matches the target grid.
        This ensures that FSS computations only include grid points within the intended
        verification domain and out-of-domain points do not dilute the skill score.

        Parameters:
            fcst (xr.DataArray): Regridded forecast array on the common target grid.
            obs (xr.DataArray): Regridded observation array on the common target grid.
            region_mask (xr.DataArray): Binary domain mask with ``latitude``/``longitude``
                coordinates.
            target_resolution (str or float): Target resolution used to resolve the mask grid.
            lat_min (float): Southern boundary of the domain in degrees.
            lat_max (float): Northern boundary of the domain in degrees.
            lon_min (float): Western boundary of the domain in degrees.
            lon_max (float): Eastern boundary of the domain in degrees.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: Tuple of ``(fcst_masked, obs_masked)`` with
                out-of-domain points set to NaN.
        """
        if target_resolution in ("obs", "fcst"):
            mask_r = region_mask.interp_like(obs, method="nearest")
        else:
            res = float(target_resolution)
            t_lats = np.arange(lat_min, lat_max + res, res)
            t_lons = np.arange(lon_min, lon_max + res, res)
            mask_r = region_mask.interp(latitude=t_lats, longitude=t_lons, method="nearest")

        mask_bool = mask_r.values > 0.5
        fcst_masked = fcst.where(mask_bool)
        obs_masked = obs.where(mask_bool)

        logger.debug(
            "Mask applied: %d domain points, fcst valid %d, obs valid %d",
            mask_bool.sum(),
            (~np.isnan(fcst_masked.values)).sum(),
            (~np.isnan(obs_masked.values)).sum(),
        )
        return fcst_masked, obs_masked

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def prepare(
        self,
        forecast_accum: xr.DataArray,
        observation_accum: xr.DataArray,
        region_mask: xr.DataArray,
        valid_time: datetime.datetime,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Execute the complete data preparation pipeline for a single valid time.
        Processing steps are applied in sequence: coordinate standardisation, mask-extent
        computation, observation clipping, regridding to the target resolution, and
        verification-mask application. A shape assertion is performed after masking to
        guarantee that forecast and observation arrays are co-located on the same grid
        before being passed to PerfMetrics.

        Parameters:
            forecast_accum (xr.DataArray): Accumulated forecast precipitation field on
                its native grid.
            observation_accum (xr.DataArray): Accumulated observation precipitation field
                on its native grid.
            region_mask (xr.DataArray): Binary verification-domain mask with standard
                coordinate names.
            valid_time (datetime.datetime): Start of the accumulation window, used for
                log messages.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: Tuple of ``(forecast_da, observation_da)``
                on the same masked grid.
        """
        cfg = self.config
        logger.debug(
            "Preparing data for %s – %s",
            valid_time.strftime("%Y-%m-%d %H:%M"),
            (valid_time + cfg.forecast_step).strftime("%Y-%m-%d %H:%M"),
        )

        lat_min, lat_max, lon_min, lon_max = self.get_mask_extent(region_mask)
        obs_prepped = self.prepare_observation_coordinates(observation_accum)
        obs_clipped = self.clip_observations(obs_prepped, lat_min, lat_max, lon_min, lon_max)

        fcst_r, obs_r = self.regrid_to_target(
            forecast_accum, obs_clipped, cfg.target_resolution,
            lat_min, lat_max, lon_min, lon_max,
        )

        fcst_masked, obs_masked = self.apply_verification_mask(
            fcst_r, obs_r, region_mask, cfg.target_resolution,
            lat_min, lat_max, lon_min, lon_max,
        )

        assert fcst_masked.shape == obs_masked.shape, (
            f"Grid mismatch: fcst {fcst_masked.shape} vs obs {obs_masked.shape}"
        )

        valid_pairs = int(
            (~np.isnan(fcst_masked.values) & ~np.isnan(obs_masked.values)).sum()
        )
        logger.debug("Valid forecast–observation pairs: %d / %d", valid_pairs, fcst_masked.size)

        return fcst_masked, obs_masked
