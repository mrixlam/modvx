#!/usr/bin/env python3

"""
Grid preparation and data validation for MODvx.

This module contains the DataValidator class, which is responsible for preparing forecast and observation fields for co-located, masked verification arrays. The preparation steps include standardizing observation coordinates, computing the geographic extent of the verification domain mask, clipping observations to a buffered extent around the domain, regridding both forecast and observation to a common target grid, and applying the verification-domain mask to ensure that only in-domain points are included in the final arrays used for verification metrics computation. The DataValidator class is designed to be flexible and configurable via the ModvxConfig object, allowing users to specify target resolutions, buffer sizes, and other parameters that influence how the data is processed. By centralizing all data preparation logic within this class, we ensure that the forecast and observation fields are consistently processed and ready for accurate skill score calculations in the PerfMetrics class. 

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
    """ Prepare forecast and observation fields for co-located, masked verification arrays. """

    def __init__(self: "DataValidator", 
                 config: ModvxConfig) -> None:
        """ 
        This class is responsible for preparing forecast and observation fields for co-located, masked verification arrays. The preparation steps include standardizing observation coordinates, computing the geographic extent of the verification domain mask, clipping observations to a buffered extent around the domain, regridding both forecast and observation to a common target grid, and applying the verification-domain mask to ensure that only in-domain points are included in the final arrays used for verification metrics computation. The DataValidator class is designed to be flexible and configurable via the ModvxConfig object, allowing users to specify target resolutions, buffer sizes, and other parameters that influence how the data is processed. By centralizing all data preparation logic within this class, we ensure that the forecast and observation fields are consistently processed and ready for accurate skill score calculations in the PerfMetrics class. 

        Parameters:
            config (ModvxConfig): Run configuration with target resolution, clip buffer size, and verbosity settings.

        Returns:
            None
        """
        # Specify the configuration for data validation and preparation. 
        self.config = config


    @staticmethod
    def standardize_observation_coordinates(obs: xr.DataArray) -> xr.DataArray:
        """
        This static method standardizes the coordinate names and longitude convention of the input observation DataArray. It first uses the standardize_coords utility function to rename any non-standard latitude and longitude coordinate names to "latitude" and "longitude", ensuring consistent processing in subsequent steps. Then, it applies the normalize_longitude function to convert longitudes to the [0, 360] convention, which is commonly used in meteorological datasets and ensures compatibility with the verification domain masks and regridding operations. The resulting standardized observation DataArray is returned for further processing in the data preparation pipeline. 

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
    def compute_mask_extent(region_mask: xr.DataArray,) -> Tuple[float, float, float, float]:
        """
        This static method computes the geographic extent of the verification domain mask by identifying the minimum and maximum latitude and longitude values that encompass all active (non-zero) grid points in the mask. It creates boolean arrays to determine which rows and columns of the mask contain active points, then extracts the corresponding latitude and longitude coordinates. If no active points are found in the mask, a ValueError is raised to indicate that there are no valid points in the verification domain. The computed bounding box is returned as a tuple of (lat_min, lat_max, lon_min, lon_max) in degrees, which is used in subsequent steps to clip observations and define the target grid for regridding. 

        Parameters:
            region_mask (xr.DataArray): Binary verification-domain mask with ``latitude``/``longitude`` coordinates.

        Returns:
            Tuple[float, float, float, float]: Bounding box as ``(lat_min, lat_max, lon_min, lon_max)`` in degrees.
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


    def clip_observation_to_buffer(self: "DataValidator",
                                   obs: xr.DataArray,
                                   lat_min: float,
                                   lat_max: float,
                                   lon_min: float,
                                   lon_max: float,) -> xr.DataArray:
        """
        This method clips the observation DataArray to a buffered extent around the verification domain defined by the input latitude and longitude bounds. The buffer size in degrees is retrieved from the configuration (``clip_buffer_deg``) and applied to expand the clipping bounds in all directions. The method ensures that the buffered bounds do not exceed valid geographic limits (i.e., latitudes between -90 and 90 degrees, longitudes between 0 and 360 degrees). The resulting clipped observation DataArray is returned for further processing, such as regridding and masking. This clipping step helps to reduce the computational load of subsequent operations by limiting the observation data to a region that encompasses the verification domain with a reasonable margin. 

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
    def regrid_to_common_grid(fcst: xr.DataArray,
                              obs_clipped: xr.DataArray,
                              target_resolution: Union[str, float],
                              lat_min: float,
                              lat_max: float,
                              lon_min: float,
                              lon_max: float,) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        This static method regrids both the forecast and clipped observation DataArrays to a common target grid defined by the specified target resolution and geographic bounds. The target resolution can be specified as "obs" to interpolate the forecast to the observation grid, "fcst" to interpolate the observation to the forecast grid, or as a float value representing the degree spacing for a new regular lat-lon grid. For the "obs" and "fcst" cases, xarray's interp_like method is used for interpolation. For a custom float resolution, new latitude and longitude arrays are created based on the provided bounds and resolution, and both forecast and observation are interpolated to this new grid using linear interpolation. The resulting regridded forecast and observation DataArrays are returned on the common target grid for subsequent masking and verification metric calculations. 

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
    def apply_domain_mask(fcst: xr.DataArray,
                          obs: xr.DataArray,
                          region_mask: xr.DataArray,
                          target_resolution: Union[str, float],
                          lat_min: float,
                          lat_max: float,
                          lon_min: float,
                          lon_max: float,) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        This static method applies the verification-domain mask to both the forecast and observation DataArrays on the common target grid. If the target resolution is "obs" or "fcst", the region mask is resampled to the corresponding grid using nearest-neighbour interpolation. For a custom float resolution, the mask is first resampled to the new regular grid defined by the provided bounds and resolution. A boolean mask is created where True indicates points inside the verification domain (mask value > 0.5). This boolean mask is then applied to both forecast and observation DataArrays using xarray's where method, setting out-of-domain points to NaN. The method logs the number of valid points remaining in both forecast and observation after masking for debugging purposes. Finally, the masked forecast and observation DataArrays are returned for use in verification metrics computation.

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


    def prepare(self: "DataValidator",
                forecast_accum: xr.DataArray,
                observation_accum: xr.DataArray,
                region_mask: xr.DataArray,
                valid_time: datetime.datetime,) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        This method orchestrates the full preparation of forecast and observation DataArrays for verification. It first computes the geographic extent of the verification domain mask, then standardizes the observation coordinates and longitude convention. Next, it clips the observation to a buffered extent around the domain to reduce data volume for subsequent steps. Both forecast and observation are then regridded to a common target grid defined by the configuration. Finally, the verification-domain mask is applied to both arrays to ensure that only in-domain points are included in the final arrays used for verification metrics computation. The method logs key steps and counts of valid points at each stage for debugging purposes, and returns the fully prepared forecast and observation DataArrays ready for skill score calculations. 

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
