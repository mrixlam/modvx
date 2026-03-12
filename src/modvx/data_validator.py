#!/usr/bin/env python3

"""
Grid preparation and data validation for MODvx.

This module defines the DataValidator class, which is responsible for preparing the MPAS grid and validating the integrity of forecast and observation data before verification. The validator ensures that the grid is properly loaded and remapped to a regular lat-lon structure, checks for the presence of required variables in the input datasets, and performs any necessary preprocessing steps to align the data with the expectations of the verification algorithms. By centralizing these validation and preparation tasks, we can catch potential issues early in the workflow and ensure that downstream components receive clean, well-structured data for analysis. The DataValidator also serves as a single point of maintenance for any future changes to data handling or grid preparation logic.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
from __future__ import annotations

import logging
import datetime
import numpy as np
import xarray as xr
from typing import Tuple, Union

from .config import ModvxConfig
from .utils import normalize_longitude, standardize_coords

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Prepare forecast and observation fields for co-located, masked verification arrays. This class encapsulates every preprocessing step between raw loaded data and the analysis-ready arrays expected by PerfMetrics. Processing steps include coordinate standardisation, longitude normalisation, mask-extent calculation, observation clipping, regridding to a common grid, and verification-mask application. All behaviour is governed by the provided ModvxConfig instance.

    Parameters:
        config (ModvxConfig): Run configuration with target resolution, clip buffer size, and verbosity settings.
    """

    def __init__(self, config: ModvxConfig) -> None:
        # Specify the configuration for data validation and preparation. 
        self.config = config


    @staticmethod
    def standardize_observation_coordinates(obs: xr.DataArray) -> xr.DataArray:
        """
        Standardise the coordinate names and longitude convention of an observation DataArray. Input observation files may use abbreviated dimension names (``lat``/``lon``) and arbitrary longitude conventions. This method applies standardize_coords to rename dimensions to ``latitude``/``longitude`` and then converts longitudes to the [0, 360] convention required by the clipping and regridding steps that follow. It is always the first step applied to raw observation data in the processing pipeline.

        Parameters:
            obs (xr.DataArray): Raw observation DataArray with potentially non-standard coordinate names.

        Returns:
            xr.DataArray: Observation DataArray with standard coordinate names and [0, 360] longitudes.
        """
        # Standardize coordinate names to 'latitude' and 'longitude' for consistent processing 
        observation_std = standardize_coords(obs)

        # Normalize longitudes to [0, 360] convention for consistent processing in subsequent steps. 
        observation_std = normalize_longitude(observation_std, "0_360")

        # Return the standardized observation DataArray for further processing 
        return observation_std


    @staticmethod
    def compute_mask_extent(
        region_mask: xr.DataArray,
    ) -> Tuple[float, float, float, float]:
        """
        Compute the geographic bounding box of active (non-zero) points in a region mask. The bounding box is determined by finding the latitude and longitude coordinates that contain at least one active mask cell, then taking the min and max of each axis. The result is used to clip the observation field to the region of interest before regridding, reducing unnecessary computation. A ValueError is raised when the mask contains no active points, indicating a misconfigured or incompatible mask file.

        Parameters:
            region_mask (xr.DataArray): Binary verification-domain mask with ``latitude``/``longitude`` coordinates.

        Returns:
            Tuple[float, float, float, float]: Bounding box as ``(lat_min, lat_max, lon_min, lon_max)`` in degrees.

        Raises:
            ValueError: If the mask contains no active (non-zero) grid points.
        """
        # Create a boolean array where True indicates active mask points (values > 0). 
        active_mask = region_mask.values > 0

        # Extract latitude coordinates corresponding to rows that have at least one active mask point.
        lat_indices = region_mask.latitude.values[active_mask.any(axis=1)]

        # Extract longitude coordinates corresponding to columns that have at least one active mask point. 
        lon_indices = region_mask.longitude.values[active_mask.any(axis=0)]

        # Raise a ValueError if there are no active points in the mask
        if len(lat_indices) == 0 or len(lon_indices) == 0:
            raise ValueError("No valid points in verification domain mask")

        # Compute the bounding box of active points in the mask.
        lat_min, lat_max = float(lat_indices.min()), float(lat_indices.max())
        lon_min, lon_max = float(lon_indices.min()), float(lon_indices.max())

        # Log the computed bounding box for debugging purposes. 
        logger.debug(
            "Mask extent: lat [%.2f, %.2f], lon [%.2f, %.2f]",
            lat_min, lat_max, lon_min, lon_max,
        )

        # Return the computed bounding box for use in observation processing
        return lat_min, lat_max, lon_min, lon_max


    def clip_observation_to_buffer(
        self,
        obs: xr.DataArray,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> xr.DataArray:
        """
        Clip an observation DataArray to the verification domain extent with an added buffer. A configurable buffer (default 1°, set via ModvxConfig.clip_buffer_deg) is added on all sides of the bounding box before slicing, ensuring that the subsequent regridding step has sufficient data near domain edges to produce accurate interpolated values. Latitude and longitude bounds are clamped to valid geographic ranges ([-90, 90] and [0, 360]). This significantly reduces data volume for regional domain verification.

        Parameters:
            obs (xr.DataArray): Global or large-domain observation DataArray with standard coordinate names.
            lat_min (float): Southern latitude boundary of the verification domain in degrees.
            lat_max (float): Northern latitude boundary of the verification domain in degrees.
            lon_min (float): Western longitude boundary in degrees (in [0, 360] convention).
            lon_max (float): Eastern longitude boundary in degrees (in [0, 360] convention).

        Returns:
            xr.DataArray: Observation DataArray clipped to the buffered domain extent.
        """
        # Retrieve the buffer size from the configuration
        buffer_deg = self.config.clip_buffer_deg

        # Clip the observation to the buffered extent, ensuring bounds are within valid geographic limits.
        observation_clipped = obs.sel(
            latitude=slice(max(lat_min - buffer_deg, -90), min(lat_max + buffer_deg, 90)),
            longitude=slice(max(lon_min - buffer_deg, 0), min(lon_max + buffer_deg, 360)),
        )

        # Log the shape of the clipped observation for debugging purposes.
        logger.debug(
            "Observation clipped: %d × %d",
            observation_clipped.latitude.size,
            observation_clipped.longitude.size,
        )

        # Return the clipped observation DataArray for further processing
        return observation_clipped


    @staticmethod
    def regrid_to_common_grid(
        fcst: xr.DataArray,
        obs_clipped: xr.DataArray,
        target_resolution: Union[str, float],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Regrid forecast and observation arrays to a common regular lat-lon grid. Three target resolution modes are supported: ``"obs"`` interpolates the forecast to match the observation grid; ``"fcst"`` interpolates the observation to the forecast grid; and a floating-point value creates a new regular grid at that degree spacing covering the domain bounds. Linear interpolation is used for floating-point targets. The returned arrays share identical coordinates.

        Parameters:
            fcst (xr.DataArray): Forecast precipitation field on its native grid.
            obs_clipped (xr.DataArray): Clipped observation DataArray on its native grid.
            target_resolution (str or float): Regrid target: ``"obs"``, ``"fcst"``, or a float degree spacing.
            lat_min (float): Southern latitude boundary of the target domain in degrees.
            lat_max (float): Northern latitude boundary of the target domain in degrees.
            lon_min (float): Western longitude boundary of the target domain in degrees.
            lon_max (float): Eastern longitude boundary of the target domain in degrees.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: Tuple of ``(fcst_regridded, obs_regridded)`` on the common grid.
        """
        # Handle the "obs" target resolution case by interpolating the forecast to the observation grid.
        if target_resolution == "obs":
            return fcst.interp_like(obs_clipped), obs_clipped

        # When target resolution is "fcst", we interpolate the observation to the forecast grid.
        if target_resolution == "fcst":
            return fcst, obs_clipped.interp_like(fcst)

        # Standardize the target resolution to a float value for grid construction.
        res = float(target_resolution)

        # Create target latitude and longitude arrays for the new regular grid 
        target_lats = np.arange(lat_min, lat_max + res, res)
        target_lons = np.arange(lon_min, lon_max + res, res)

        # Interpolate the forecast to the target grid using linear interpolation.
        fcst_regridded = fcst.interp(latitude=target_lats, longitude=target_lons, method="linear")

        # Interpolate the clipped observation to the same target grid using linear interpolation.
        obs_regridded = obs_clipped.interp(latitude=target_lats, longitude=target_lons, method="linear")

        # Return the regridded forecast and observation DataArrays on the common target grid 
        return fcst_regridded, obs_regridded


    @staticmethod
    def apply_domain_mask(
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
        Apply a binary verification-domain mask to forecast and observation arrays. Grid points outside the mask (where mask value ≤ 0.5 after interpolation) are set to NaN in both arrays. The mask is first resampled to the data grid via nearest-neighbour interpolation so that its resolution matches the target grid. This ensures that FSS computations only include grid points within the intended verification domain and out-of-domain points do not dilute the skill score.

        Parameters:
            fcst (xr.DataArray): Regridded forecast array on the common target grid.
            obs (xr.DataArray): Regridded observation array on the common target grid.
            region_mask (xr.DataArray): Binary domain mask with ``latitude``/``longitude`` coordinates.
            target_resolution (str or float): Target resolution used to resolve the mask grid.
            lat_min (float): Southern boundary of the domain in degrees.
            lat_max (float): Northern boundary of the domain in degrees.
            lon_min (float): Western boundary of the domain in degrees.
            lon_max (float): Eastern boundary of the domain in degrees.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: Tuple of ``(fcst_masked, obs_masked)`` with out-of-domain points set to NaN.
        """
        if target_resolution in ("obs", "fcst"):
            # Resample the region mask to the target grid using nearest-neighbour interpolation. 
            mask_resampled = region_mask.interp_like(obs, method="nearest")
        else:
            # Standardize the target resolution to a float value for grid construction.
            res = float(target_resolution)

            # Create target latitude and longitude arrays for resampling the mask to the target grid.
            t_lats = np.arange(lat_min, lat_max + res, res)
            t_lons = np.arange(lon_min, lon_max + res, res)

            # Resample the region mask to the target grid using nearest-neighbour interpolation. 
            mask_resampled = region_mask.interp(latitude=t_lats, longitude=t_lons, method="nearest")

        # Create a boolean mask where True indicates points inside the verification domain (mask value > 0.5).
        mask_bool = mask_resampled.values > 0.5

        # Apply the mask to both forecast and observation arrays
        forecast_masked = fcst.where(mask_bool)
        observation_masked = obs.where(mask_bool)

        # Log the number of valid points after masking for debugging purposes.
        logger.debug(
            "Mask applied: %d domain points, fcst valid %d, obs valid %d",
            mask_bool.sum(),
            (~np.isnan(forecast_masked.values)).sum(),
            (~np.isnan(observation_masked.values)).sum(),
        )

        # Return the masked forecast and observation DataArrays for use in verification metrics computation
        return forecast_masked, observation_masked


    def prepare(
        self,
        forecast_accum: xr.DataArray,
        observation_accum: xr.DataArray,
        region_mask: xr.DataArray,
        valid_time: datetime.datetime,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Execute the complete data preparation pipeline for a single valid time. Processing steps are applied in sequence: coordinate standardisation, mask-extent computation, observation clipping, regridding to the target resolution, and verification-mask application. A shape assertion is performed after masking to guarantee that forecast and observation arrays are co-located on the same grid before being passed to PerfMetrics.

        Parameters:
            forecast_accum (xr.DataArray): Accumulated forecast precipitation field on its native grid.
            observation_accum (xr.DataArray): Accumulated observation precipitation field on its native grid.
            region_mask (xr.DataArray): Binary verification-domain mask with standard coordinate names.
            valid_time (datetime.datetime): Start of the accumulation window, used for log messages.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: Tuple of ``(forecast_da, observation_da)`` on the same masked grid.
        """
        # Retrieve the configuration for use in processing steps.
        config = self.config

        # Log the valid time and accumulation window for debugging purposes. 
        logger.debug(
            "Preparing data for %s – %s",
            valid_time.strftime("%Y-%m-%d %H:%M"),
            (valid_time + config.forecast_step).strftime("%Y-%m-%d %H:%M"),
        )

        # Compute the geographic extent of the verification domain mask 
        lat_min, lat_max, lon_min, lon_max = self.compute_mask_extent(region_mask)

        # Standardize the observation coordinates and longitude convention before any spatial operations. 
        observation_standardized = self.standardize_observation_coordinates(observation_accum)

        # Clip the observation to the buffered extent around the verification domain 
        observation_clipped = self.clip_observation_to_buffer(
            observation_standardized, lat_min, lat_max, lon_min, lon_max
        )

        # Regrid both forecast and observation to the common target grid defined by the configuration
        fcst_regridded, obs_regridded = self.regrid_to_common_grid(
            forecast_accum, observation_clipped, config.target_resolution,
            lat_min, lat_max, lon_min, lon_max,
        )

        # Apply the verification-domain mask to both forecast and observation arrays
        forecast_masked, observation_masked = self.apply_domain_mask(
            fcst_regridded, obs_regridded, region_mask, config.target_resolution,
            lat_min, lat_max, lon_min, lon_max,
        )

        # Verify that the masked forecast and observation arrays have the same shape
        assert forecast_masked.shape == observation_masked.shape, (
            f"Grid mismatch: fcst {forecast_masked.shape} vs obs {observation_masked.shape}"
        )

        # Count the number of valid (non-NaN) forecast-observation pairs after masking for logging purposes. 
        valid_pairs = int((~np.isnan(forecast_masked.values) & ~np.isnan(observation_masked.values)).sum())

        # Log the number of valid forecast-observation pairs after preparation for debugging purposes.
        logger.debug("Valid forecast–observation pairs: %d / %d", valid_pairs, forecast_masked.size)

        # Return the fully prepared forecast and observation DataArrays for verification metrics computation
        return forecast_masked, observation_masked
