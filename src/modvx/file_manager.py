#!/usr/bin/env python3

"""
File I/O handling for MODvx.

This module contains the FileManager class, which centralizes all filesystem operations for a modvx verification run. The FileManager is responsible for constructing paths to forecast and observation files based on the provided configuration and temporal identifiers, loading verification-domain masks, and managing an in-memory and disk-based cache for intermediate results such as remapped forecast accumulations. By encapsulating all file-related logic within this class, we ensure a clean separation of concerns and provide a single point of maintenance for any future changes to file handling or caching strategies. The FileManager also includes utility methods for grouping observation times by their corresponding daily files and mapping observation end-times to their appropriate indices within those files, facilitating efficient loading of observation data for multi-hour accumulation windows.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
from __future__ import annotations

import os
import logging
import datetime
import xarray as xr
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .config import ModvxConfig
from .utils import format_threshold_for_filename, normalize_longitude, standardize_coords

logger = logging.getLogger(__name__)

# Metric keys for output files and CSV extraction
_FSS_METRIC_KEYS: list[str] = ["fss"]
_CONTINGENCY_KEYS: list[str] = ["pod", "far", "csi", "fbias", "ets"]
_ALL_METRIC_KEYS: list[str] = ["fss", "pod", "far", "csi", "fbias", "ets"]


class FileManager:
    """ Centralise all filesystem operations for a modvx verification run. """

    def __init__(self: "FileManager", 
                 config: ModvxConfig) -> None:
        """
        This initializer sets up the FileManager with the necessary configuration for file handling and caching. The provided ModvxConfig object contains all relevant settings such as directory paths, filename templates, and variable names, which will be used in constructing paths to forecast and observation files, loading masks, and managing caches. The initializer also initializes in-memory cache dictionaries for observations and forecasts, and ensures that the shared disk cache directory exists if it is configured. By storing the configuration as an instance variable, we ensure that all methods within the FileManager have access to consistent parameters for constructing file paths and managing caches according to the user's specifications. 

        Parameters:
            config (ModvxConfig): Run configuration with directory paths, filename templates, and variable names.

        Returns:
            None
        """
        # Store the configuration for use in path construction, loading, and caching operations
        self.config = config

        # Initialize the in-memory observation cache dict
        self._obs_mem_cache: Dict[str, xr.DataArray] = {}

        # Initialize the in-memory forecast cache dict
        self._fcst_mem_cache: Dict[str, xr.DataArray] = {}

        # Ensure the shared disk cache directory exists for multiprocessing workers 
        if self.config.cache_dir is None:
            self.config.cache_dir = os.path.join(
                config.resolve_relative_path(config.output_dir), ".obs_cache",
            )


    def get_forecast_filepath(self: "FileManager",
                              valid_time: datetime.datetime,
                              init_string: str,) -> str:
        """
        This method constructs the full path to an MPAS diag NetCDF file for a given forecast valid time and cycle initialisation string. The path is built based on the configured directory structure and filename template, which typically includes the experiment name, cycle information, and a timestamp derived from the valid time. The method formats the valid time into a string suitable for the filename and combines it with the initialisation string to produce the final path. This allows for consistent and flexible access to forecast files across different experiments and cycles without hardcoding specific paths. 

        Parameters:
            valid_time (datetime.datetime): Forecast valid time, used to format the filename.
            init_string (str): Cycle initialisation string in ``YYYYmmddHH`` format (e.g. ``"2024091700"``).

        Returns:
            str: Full path to the MPAS diag NetCDF file for the specified valid time and cycle.
        """
        # Retrieve the configuration for use in path construction.
        config = self.config

        # Extract the timestamp string from the valid_time for filename formatting
        timestamp_str = valid_time.strftime("%Y-%m-%d_%H.%M.%S")

        # Return the constructed path to the MPAS diagnostic file based on the configured directory structure and filename template. 
        return str(
            Path(config.resolve_relative_path(config.fcst_dir))
            / config.experiment_name
            / "ExtendedFC"
            / init_string
            / f"diag.{timestamp_str}.nc"
        )


    def get_observation_filepath(self: "FileManager", 
                                 date_key: str) -> str:
        """
        This method constructs the full path to a FIMERG daily NetCDF file for a given date key in ``YYYYmmdd`` format. It resolves the base observation directory from the configuration and iterates through the list of vintage preferences to check for the existence of files corresponding to each vintage. The method returns the path to the first existing file it finds, allowing for flexible handling of multiple vintages without hardcoding specific paths. If no files are found for any vintage, it returns the path corresponding to the first vintage in the preference list, which will likely result in a FileNotFoundError when accessed, signaling that the expected observation file is missing. 

        Parameters:
            date_key (str): Date string in ``YYYYmmdd`` format (e.g. ``"20240917"``).

        Returns:
            str: Full path to the FIMERG daily NetCDF file, using the first available vintage.
        """
        # Resolve the base observation directory from the configuration
        obs_dir = self.config.resolve_relative_path(self.config.obs_dir)

        # Iterate through the vintage preference list and return the path to the first existing file.
        for vintage in self.config.obs_vintage_preference:
            path = (
                f"{obs_dir}/IMERG.A01H.VLD{date_key}.S{date_key}T000000."
                f"E{date_key}T235959.{vintage}.V07B.SRCHHR.X360Y180.R1p0.FMT.nc"
            )

            # If the path exists, return it immediately
            if os.path.exists(path):
                return path
        
        # Return the full path to the observation file corresponding to the first vintage in the preference list
        return (
            f"{obs_dir}/IMERG.A01H.VLD{date_key}.S{date_key}T000000."
            f"E{date_key}T235959.{self.config.obs_vintage_preference[0]}.V07B.SRCHHR.X360Y180.R1p0.FMT.nc"
        )


    @staticmethod
    def get_observation_hour_index(time: datetime.datetime) -> int:
        """
        This static method maps an observation end-time to the corresponding zero-based time index (0–23) used in the FIMERG daily NetCDF files. FIMERG files use an end-of-hour timestamp convention, where the value at hour 0 (00:00 UTC) corresponds to the accumulation from the previous day, and values at hours 1–23 correspond to accumulations ending at those hours on the same day. Therefore, if the input time is at midnight (00:00), the method returns index 23 to access the previous day's final accumulation. For all other hours, it returns time.hour - 1 to align with FIMERG's indexing. This mapping is essential for correctly extracting hourly observations from the daily files based on their end-times. 

        Parameters:
            time (datetime.datetime): Observation end-time to convert to a file index.

        Returns:
            int: Zero-based time index (0–23) into the FIMERG daily NetCDF file.
        """
        # Return index 23 for midnight (00:00) and time.hour - 1 for all other hours 
        return 23 if time.hour == 0 else time.hour - 1


    @staticmethod
    def group_observation_times_by_date(start_time: datetime.datetime,
                                        end_time: datetime.datetime,
                                        obs_interval: datetime.timedelta,) -> Dict[str, List[datetime.datetime]]:
        """
        This static method groups a sequence of observation end-times by their corresponding daily FIMERG file. It iterates from *start_time* to *end_time* inclusive at *obs_interval* steps, assigning each time to the appropriate date key. Times at midnight (00:00 UTC) are assigned to the previous calendar day's file, consistent with FIMERG's end-of-hour timestamp convention. The result allows loading each daily file exactly once when accumulating observations over a multi-hour window.

        Parameters:
            start_time (datetime.datetime): First observation end-time to include (inclusive).
            end_time (datetime.datetime): Last observation end-time to include (inclusive).
            obs_interval (datetime.timedelta): Time step between consecutive observations.

        Returns:
            Dict[str, List[datetime.datetime]]: Mapping of ``YYYYmmdd`` date keys to lists of datetime objects belonging to that file.
        """
        # Initialize a defaultdict to hold lists of observation times keyed by date strings
        times_by_date: Dict[str, List[datetime.datetime]] = defaultdict(list)

        # Set the current time to the start time for iteration
        current = start_time

        # Loop through the observation times from start to end inclusive
        while current <= end_time:
            if current.hour == 0:
                # For midnight times, assign to the previous day's file 
                date_key = (current - datetime.timedelta(days=1)).strftime("%Y%m%d")
            else:
                # For all other times, assign to the current day's file
                date_key = current.strftime("%Y%m%d")
            
            # Append the current time to the list for the appropriate date key
            times_by_date[date_key].append(current)

            # Advance the current time by the observation interval for the next iteration
            current += obs_interval
        
        # Return the mapping of date keys to lists of observation end-times 
        return times_by_date


    def load_region_mask(self: "FileManager", 
                         mask_filepath: str) -> Tuple[xr.DataArray, str]:
        """
        This method loads a verification-domain mask from a specified NetCDF file path. It first checks if the file exists and raises a FileNotFoundError if it does not. The method then reads the dataset using xarray and identifies the first non-coordinate variable to use as the mask variable. It standardizes the coordinate names to latitude and longitude, normalizes longitudes to the [0, 360] convention, and logs the number of valid (non-zero) points in the mask for debugging purposes. Finally, it returns the mask as an xarray DataArray along with the variable name used for the mask. This allows for flexible handling of different mask files while ensuring that the resulting mask is in a consistent format for subsequent processing steps. 

        Parameters:
            mask_filepath (str): Full path to the mask NetCDF file.

        Returns:
            Tuple[xr.DataArray, str]: Tuple of ``(mask_array, variable_name)`` where the mask has standard coordinate names and [0, 360] longitudes.
        """
        # Raise a FileNotFoundError if the specified mask file does not exist on disk 
        if not os.path.exists(mask_filepath):
            raise FileNotFoundError(f"Mask file not found: {mask_filepath}")

        # Read the mask dataset with xarray
        ds = xr.open_dataset(mask_filepath)

        # Identify non-coordinate variables by excluding common coordinate names. 
        data_vars = [
            v for v in ds.data_vars
            if v not in {"lat", "lon", "latitude", "longitude"}
        ]

        # Raise a ValueError if no non-coordinate variable is found 
        if not data_vars:
            raise ValueError(f"No mask variable found in {mask_filepath}")

        # Use the first non-coordinate variable as the mask variable
        var_name = str(data_vars[0])

        # Standardize coordinate names to latitude/longitude
        mask = standardize_coords(ds[var_name])

        # Normalize longitudes to [0, 360] convention for consistency with observation data
        mask = normalize_longitude(mask, "0_360")

        # Log the number of valid (non-zero) points in the mask for debugging purposes.
        logger.debug(
            "Loaded mask '%s' from %s — %d/%d valid (%.1f%%)",
            var_name,
            mask_filepath,
            (mask.values > 0).sum(),
            mask.size,
            100 * (mask.values > 0).sum() / mask.size,
        )

        # Return the mask DataArray and the variable name
        return mask, var_name


    def _forecast_cache_key(self: "FileManager", 
                            init_string: str, 
                            valid_time: datetime.datetime,) -> str:
        """
        This internal helper method generates a deterministic cache key string for storing and retrieving accumulated forecast DataArrays based on the cycle initialisation string, valid time, and forecast step duration. The key is constructed in the format ``"fcst_accum_<expname>_<init>_<validtime>_<N>h"`` where <expname> is the experiment name from the configuration, <init> is the cycle initialisation string, <validtime> is the valid time formatted as ``YYYYmmddHH``, and <N> is the forecast step duration in hours. This consistent key format allows for efficient caching of intermediate results across different runs and processes while ensuring that keys are unique to specific forecast accumulations. 

        Parameters:
            init_string (str): Cycle initialisation string in ``YYYYmmddHH`` format.
            valid_time (datetime.datetime): Start of the accumulation window.

        Returns:
            str: Deterministic cache key string in the format ``"fcst_accum_<expname>_<init>_<validtime>_<N>h"``.
        """
        # Calculate the forecast step duration in hours from the configuration's forecast_step timedelta
        step_h = int(self.config.forecast_step.total_seconds() / 3600)

        # Extract experiment name from the configuration 
        experiment_name = self.config.experiment_name

        # Construct and return the cache key string using the experiment name, initialization string, valid time, and forecast step.
        return f"fcst_accum_{experiment_name}_{init_string}_{valid_time.strftime('%Y%m%d%H')}_{step_h}h"


    def _load_cache_entry(self: "FileManager", 
                          key: str, 
                          in_memory_cache: Dict[str, xr.DataArray],) -> Optional[xr.DataArray]:
        """
        This internal helper method attempts to load a cached DataArray from either the in-memory cache dict or the shared disk cache directory based on the provided key. It first checks the in-memory cache for the key and returns the cached DataArray if found. If not found in memory, it constructs the expected disk cache file path using the key and checks if it exists. If a disk cache file is found, it loads the dataset using xarray, extracts the first data variable as the cached DataArray, stores it in the in-memory cache for future access, and returns it. If the key is not found in either cache, it returns None to indicate a cache miss. This method allows for efficient retrieval of intermediate results while minimizing redundant computations across processes. 

        Parameters:
            key (str): Cache key string to lookup.
            in_memory_cache (Dict[str, xarray.DataArray]): In-memory cache mapping.

        Returns:
            Optional[xarray.DataArray]: Cached DataArray if found, otherwise ``None``.
        """
        # First check the in-memory cache dict for the key and return if found
        if key in in_memory_cache:
            logger.debug("Cache hit (memory): %s", key)
            return in_memory_cache[key]

        # Set the cache directory from the configuration for use in disk cache lookup
        cache_dir = self.config.cache_dir

        if cache_dir:
            # Check for the existence of a disk cache file corresponding to the key 
            disk_path = os.path.join(cache_dir, f"{key}.nc")

            if os.path.exists(disk_path):
                # Log the disk cache hit for debugging purposes
                logger.debug("Cache hit (disk): %s", disk_path)

                # Load the dataset from the disk cache file using xarray
                dataset = xr.load_dataset(disk_path)

                # Assume the first data variable in the file is the cached DataArray
                data_array = dataset[list(dataset.data_vars)[0]]

                # Store the loaded DataArray in the in-memory cache for future quick access
                in_memory_cache[key] = data_array

                # Return the loaded DataArray from disk cache
                return data_array
        
        # Return None if the key is not found 
        return None


    def _save_cache_entry(self: "FileManager", 
                          key: str, 
                          data_array: xr.DataArray, 
                          in_memory_cache: Dict[str, xr.DataArray],) -> None:
        """
        This internal helper method saves a DataArray to both the in-memory cache dict and the shared disk cache directory under the provided key. It first stores the DataArray in the in-memory cache for quick access during the current run. If a disk cache directory is configured, it ensures that the directory exists, constructs a temporary file path for atomic writing, saves the DataArray to a NetCDF file at the temporary path, and then moves it to the final disk cache path corresponding to the key. This approach minimizes the risk of corrupted cache files due to concurrent writes and allows for efficient caching of intermediate results across processes. The method logs the cache save operation for debugging purposes but does not return any value.

        Parameters:
            key (str): Cache key under which to store the DataArray.
            data_array (xarray.DataArray): DataArray to persist.
            in_memory_cache (Dict[str, xarray.DataArray]): In-memory cache mapping.

        Returns:
            None
        """
        # Store the DataArray in the in-memory cache dict under the provided key
        in_memory_cache[key] = data_array

        # Set the cache directory from the configuration for use in disk cache lookup
        cache_dir = self.config.cache_dir

        if cache_dir:
            # Create the cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)

            # Construct the full path to the disk cache file for this key
            disk_path = os.path.join(cache_dir, f"{key}.nc")

            # Specify a temporary file path for atomic write 
            tmp_path = f"{disk_path}.tmp.{os.getpid()}"

            # Write the DataArray to a temporary NetCDF file first 
            data_array.to_netcdf(tmp_path)

            # Move the temporary file to the final disk cache path 
            os.rename(tmp_path, disk_path) 

            # Log the cache save operation for debugging purposes
            logger.debug("Cached to disk: %s", disk_path)


    def _compute_forecast_accumulation(self: "FileManager", 
                                       valid_time: datetime.datetime, 
                                       init_string: str,) -> xr.DataArray:
        """
        This internal helper method computes the accumulated precipitation for a single forecast step by loading the cumulative precipitation fields from the start and end of the accumulation window from MPAS diag files, calculating the difference to get the accumulation over the forecast step, and remapping the result from the native MPAS mesh to a regular lat-lon grid at the configured resolution. It retrieves the necessary configuration parameters for path construction and remapping, resolves the grid file path for loading coordinate information, calculates the appropriate start and end file paths based on the valid time and forecast step, and logs the files being used for debugging. After loading the start and end precipitation fields, it computes the accumulation by subtraction, updates metadata, performs remapping using the mpasdiag library, and logs the successful computation before returning the remapped accumulated precipitation as an xarray DataArray. 

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window.
            init_string (str): Cycle initialisation string in ``YYYYmmddHH`` format.

        Returns:
            xarray.DataArray: Accumulated precipitation remapped to a lat-lon grid in millimetres.
        """
        from .mpas_reader import load_mpas_precip, remap_to_latlon

        # Retrieve the configuration for use in processing steps.
        config = self.config

        # Resolve the path to the MPAS grid file for loading coordinate information needed for remapping
        grid_file = config.resolve_relative_path(config.mpas_grid_file)

        # Calculate the end of the accumulation window by adding the forecast step to the valid time
        window_end = valid_time + config.forecast_step

        # Specify the end file based on the window end time
        end_file = self.get_forecast_filepath(window_end, init_string)

        # Specify the start file based on the forecast step duration. 
        start_file = self.get_forecast_filepath(valid_time, init_string)

        # Log the start and end files being used for the accumulation computation 
        logger.debug("MPAS accum: %s → %s", start_file, end_file)

        # Load the end precipitation field, which is cumulative from model initialisation
        end_precip = load_mpas_precip(end_file, grid_file)

        # Load the start precipitation field, which is cumulative from model initialisation
        start_precip = load_mpas_precip(start_file, grid_file)

        # Compute the accumulated precipitation over the forecast step 
        accum_mesh = end_precip - start_precip

        # Clean up the loaded start and end precipitation meshes to free memory
        del end_precip, start_precip

        # Update metadata for the accumulated precipitation field
        accum_mesh.attrs["units"] = "mm"
        accum_mesh.attrs["long_name"] = f"{int(config.forecast_step_hours)}h accumulated precipitation"

        # Remap the accumulated precipitation from the native MPAS mesh to a regular lat-lon grid 
        remapped_da = remap_to_latlon(accum_mesh, end_file, grid_file, config.mpas_remap_resolution)

        # Clean up the original accumulated mesh to free memory
        del accum_mesh

        # Log the successful computation and remapping of the accumulated precipitation 
        logger.debug(
            "MPAS accumulated %dh precip from %s, remapped to %.3f° grid",
            config.forecast_step_hours, valid_time, config.mpas_remap_resolution,
        )

        # Return the remapped accumulated precipitation DataArray 
        return remapped_da


    def accumulate_forecasts(self: "FileManager",
                             valid_time: datetime.datetime,
                             init_string: str,) -> xr.DataArray:
        """
        This method computes the accumulated precipitation for a single forecast step by first checking the in-memory and disk caches for a pre-computed result using a key that encodes the valid time and forecast step. If a cached result is found, it is returned immediately. If not found in cache, it calls the internal helper method to compute the accumulation from MPAS diag files, saves the result to both caches for future access, and returns the computed DataArray. This method serves as the main entry point for obtaining forecast accumulations while leveraging caching to optimize performance across runs and processes. 

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window (window end = valid_time + forecast_step).
            init_string (str): Cycle initialisation string in ``YYYYmmddHH`` format (e.g. ``"2024091700"``).

        Returns:
            xr.DataArray: Accumulated precipitation on a regular lat-lon grid in millimetres.
        """
        # Generate a cache key for this forecast accumulation 
        key = self._forecast_cache_key(init_string, valid_time)

        # First check the in-memory cache for the accumulated precipitation using the generated key
        cached = self._load_cache_entry(key, self._fcst_mem_cache)

        # If a cached result is found in either the in-memory or disk cache, return it immediately 
        if cached is not None:
            return cached

        # Compute the accumulated precipitation from MPAS diag files without any caching
        remapped = self._compute_forecast_accumulation(valid_time, init_string)

        # Save the remapped accumulated precipitation to both the in-memory and disk caches 
        self._save_cache_entry(key, remapped, self._fcst_mem_cache)

        # Return the remapped accumulated precipitation DataArray
        return remapped


    def accumulate_forecasts_precip_accum(self: "FileManager",
                                          valid_time: datetime.datetime,
                                          init_string: str,) -> xr.DataArray:
        """
        This method computes the accumulated precipitation over the effective accumulation period defined in the configuration by summing multiple single-step accumulations if necessary. If the effective accumulation period equals the forecast step, it delegates directly to the single-step accumulation method. Otherwise, it generates a cache key for the full-window accumulation and checks both caches for a pre-computed result. If found, it returns the cached result immediately. If not found, it calculates the number of forecast steps to accumulate based on the effective accumulation hours and forecast step hours, iteratively computes each sub-step accumulation using the existing method, sums them to get the total accumulation, updates metadata, saves the total to both caches, and returns it. This method allows for flexible handling of multi-hour accumulations while maximizing cache reuse of single-step results. 

        Parameters:
            valid_time (datetime.datetime): Start of the multi-step accumulation window.
            init_string (str): Cycle initialisation string in ``YYYYmmddHH`` format.

        Returns:
            xr.DataArray: Accumulated precipitation over the full accumulation period in millimetres.
        """
        # Retrieve the configuration for use in processing steps.
        config = self.config

        # Extract the effective accumulation period in hours from configuration
        accum_h = config.effective_precip_accum_hours

        # Extract the forecast step in hours from configuration
        step_h = config.forecast_step_hours

        # If the accumulation period equals the forecast step, delegate directly to the single-step accumulation method 
        if accum_h == step_h:
            return self.accumulate_forecasts(valid_time, init_string)

        # Generate a cache key for the full-window accumulation 
        key = (
            f"fcst_accum_{config.experiment_name}_{init_string}_"
            f"{valid_time.strftime('%Y%m%d%H')}_{accum_h}h"
        )

        # Check the in-memory cache for the full-window accumulated precipitation using the generated key
        cached = self._load_cache_entry(key, self._fcst_mem_cache)

        # If a cached result is found in either the in-memory or disk cache, return it immediately
        if cached is not None:
            return cached

        # Calculate the number of forecast steps to accumulate
        n_steps = accum_h // step_h

        # Initialize an empty variable to hold the total accumulated precipitation 
        total: Optional[xr.DataArray] = None

        # Loop through each forecast step and accumulate total precipitation 
        for i in range(n_steps):
            # Calculate the valid time for this sub-step
            sub_valid = valid_time + datetime.timedelta(hours=i * step_h)

            # Compute the accumulated precipitation for this sub-step using the existing method
            step_da = self.accumulate_forecasts(sub_valid, init_string)

            # Extract total by summing the step accumulation into it
            total = step_da if total is None else total + step_da

        # Ensure that at least one forecast sub-step was accumulated
        assert total is not None, "No forecast sub-steps accumulated"

        # Update metadata for the total accumulated precipitation field
        total.attrs["units"] = "mm"
        total.attrs["long_name"] = f"{accum_h}h accumulated precipitation"

        # Save the total accumulated precipitation to both the in-memory and disk caches 
        self._save_cache_entry(key, total, self._fcst_mem_cache)

        # Log the successful accumulation of multiple forecast steps for debugging purposes.
        logger.debug(
            "Accumulated %dh forecast from %d × %dh sub-steps at %s",
            accum_h, n_steps, step_h, valid_time,
        )

        # Return total accumulated precipitation
        return total


    def _observation_cache_key(self: "FileManager", 
                               valid_time: datetime.datetime) -> str:
        """
        This internal helper method generates a deterministic cache key string for storing and retrieving accumulated observation DataArrays based on the valid time and forecast step duration. The key is constructed in the format ``"obs_accum_<YYYYmmddHH>_<N>h"`` where <YYYYmmddHH> is the valid time formatted as ``YYYYmmddHH`` and <N> is the forecast step duration in hours. This consistent key format allows for efficient caching of intermediate results across different runs and processes while ensuring that keys are unique to specific observation accumulations. 

        Parameters:
            valid_time (datetime.datetime): Start of the observation accumulation window.

        Returns:
            str: Cache key string in the format ``"obs_accum_<YYYYmmddHH>_<N>h"``.
        """
        # Calculate the forecast step in hours from the configuration's forecast_step timedelta
        step_h = int(self.config.forecast_step.total_seconds() / 3600)

        # Return a formatted cache key string for the observation accumulation 
        return f"obs_accum_{valid_time.strftime('%Y%m%d%H')}_{step_h}h"


    def accumulate_observations(self: "FileManager",
                                valid_time: datetime.datetime,) -> xr.DataArray:
        """
        This method loads and accumulates FIMERG observation data over a single forecast step by first checking the in-memory and disk caches for a pre-computed result using a key that encodes the valid time and forecast step. If a cached result is found, it is returned immediately. If not found in cache, it calls the internal helper method to compute the accumulation from FIMERG files, saves the result to both caches for future access, and returns the computed DataArray. This method serves as the main entry point for obtaining single-step observation accumulations while leveraging caching to optimize performance across runs and processes. 

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window, aligned with the forecast valid time.

        Returns:
            xr.DataArray: Accumulated hourly precipitation sum over the forecast step period.
        """
        # Generate a cache key for this observation accumulation 
        key = self._observation_cache_key(valid_time)

        # First check the in-memory cache for the accumulated observation using the generated key
        cached = self._load_cache_entry(key, self._obs_mem_cache)

        # If a cached result is found in either the in-memory or disk cache, return it immediately 
        if cached is not None:
            return cached

        # Compute the accumulated observation DataArray when no cached result is available.
        accumulated_da = self._compute_observation_accumulation_raw(valid_time)

        # Save the accumulated observation DataArray to both the in-memory and disk caches 
        self._save_cache_entry(key, accumulated_da, self._obs_mem_cache)

        # Return the accumulated observation DataArray
        return accumulated_da


    def accumulate_observations_precip_accum(self: "FileManager",
                                             valid_time: datetime.datetime,) -> xr.DataArray:
        """
        This method computes the accumulated precipitation from FIMERG observations over the effective accumulation period defined in the configuration by summing multiple single-step accumulations if necessary. If the effective accumulation period equals the forecast step, it delegates directly to the single-step accumulation method. Otherwise, it generates a cache key for the full-window accumulation and checks both caches for a pre-computed result. If found, it returns the cached result immediately. If not found, it calculates the start and end times of the accumulation window based on the valid time and configuration parameters, groups the required observation end-times by their corresponding daily files to minimize file I/O, iteratively loads and sums the relevant hourly slices from each daily file, asserts that at least one slice was accumulated to catch data gaps, updates metadata, saves the total to both caches, and returns it. This method allows for flexible handling of multi-hour observation accumulations while maximizing cache reuse of single-step results. 

        Parameters:
            valid_time (datetime.datetime): Start of the observation accumulation window.

        Returns:
            xr.DataArray: Accumulated hourly precipitation sum over the full accumulation period.
        """
        # Retrieve the configuration for use in processing steps.
        config = self.config

        # Extract the effective accumulation period in hours from configuration
        accum_h = config.effective_precip_accum_hours

        # Extract the forecast step in hours from configuration 
        step_h = config.forecast_step_hours

        # If the accumulation period equals the forecast step, delegate directly to the single-step accumulation method
        if accum_h == step_h:
            return self.accumulate_observations(valid_time)

        # Generate a cache key for the full-window observation accumulation using the valid time and accumulation period
        key = f"obs_accum_{valid_time.strftime('%Y%m%d%H')}_{accum_h}h"

        # First check the in-memory cache for the full-window accumulated observation using the generated key
        cached = self._load_cache_entry(key, self._obs_mem_cache)

        # If a cached result is found in either the in-memory or disk cache, return it immediately
        if cached is not None:
            return cached

        # Calculate the start time of the accumulation window
        start_time = valid_time + config.observation_interval

        # Calculate the end time of the accumulation window
        end_time = valid_time + config.precip_accum

        # Group the observation end-times by their corresponding daily FIMERG file 
        times_by_date = self.group_observation_times_by_date(
            start_time, end_time, config.observation_interval
        )

        # Initialize an empty variable to hold the accumulated DataArray
        accumulated_da: Optional[xr.DataArray] = None

        # Initialize a counter to track the number of hourly slices accumulated 
        count = 0

        # Loop through the grouped observation times by date 
        for date_key, times in sorted(times_by_date.items()):
            # Resolve the path to the daily FIMERG file for this date key
            daily_file = self.get_observation_filepath(date_key)

            # Log the daily file being loaded for debugging purposes
            logger.debug("Loading obs file: %s", daily_file)

            # Read the daily dataset from the resolved file path using xarray
            daily_dataset = xr.load_dataset(daily_file)

            # Loop through the observation end-times for this date key
            for obs_time in times:
                # Get the zero-based hour index in the daily observattion file
                index = self.get_observation_hour_index(obs_time)

                # Extract the hourly slice corresponding to the observation end-time index
                slice_da = daily_dataset[config.obs_var_name].isel(time=index)

                # Accumulate the hourly slice into the total, initializing if this is the first slice
                accumulated_da = slice_da if accumulated_da is None else accumulated_da + slice_da

                # Increase the counter for each hourly slice accumulated
                count += 1

            # Close the daily dataset to free memory
            daily_dataset.close()

        # Assert that at least one observation slice was accumulated to catch silent data gaps early.
        assert accumulated_da is not None, "No observation data for accumulation"

        # Log the number of hourly slices accumulated for debugging purposes.
        logger.debug("Accumulated %d obs hours (%dh window) from %s", count, accum_h, valid_time)

        # Save the accumulated observation DataArray to both the in-memory and disk caches
        self._save_cache_entry(key, accumulated_da, self._obs_mem_cache)

        # Return the accumulated observation DataArray
        return accumulated_da


    def _compute_observation_accumulation_raw(self: "FileManager",
                                              valid_time: datetime.datetime,) -> xr.DataArray:
        """
        This internal helper method computes the accumulated precipitation from FIMERG observations over a single forecast step by loading the relevant hourly slices from the daily NetCDF files based on the valid time and summing them. It calculates the start and end times of the accumulation window, groups the required observation end-times by their corresponding daily files to minimize file I/O, iteratively loads and sums the relevant hourly slices from each daily file, asserts that at least one slice was accumulated to catch data gaps, logs the number of slices accumulated for debugging purposes, and returns the total accumulated observation as an xarray DataArray. This method does not perform any caching and serves as the core computation for obtaining raw observation accumulations when no cached result is available. 

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window (window end = valid_time + forecast_step).

        Returns:
            xr.DataArray: Accumulated hourly precipitation sum over the forecast step period.
        """
        # Retrieve the configuration for use in processing steps.
        config = self.config

        # Calculate the start time of the accumulation window 
        start_time = valid_time + config.observation_interval

        # Calculate the end time of the accumulation window 
        end_time = valid_time + config.forecast_step

        # Group the observation end-times by their corresponding daily FIMERG file 
        times_by_date = self.group_observation_times_by_date(
            start_time, end_time, config.observation_interval
        )

        # Initialize an empty variable to hold the accumulated DataArray
        accumulated_da: Optional[xr.DataArray] = None

        # Initialize a counter to track the number of hourly slices accumulated 
        count = 0

        # Loop through the grouped observation times by date
        for date_key, times in sorted(times_by_date.items()):
            # Resolve the path to the daily FIMERG file for this date key
            daily_file = self.get_observation_filepath(date_key)

            # Log the daily file being loaded for debugging purposes
            logger.debug("Loading obs file: %s", daily_file)

            # Read the daily dataset from the resolved file path using xarray
            daily_dataset = xr.load_dataset(daily_file)

            # Loop through the observation end-times
            for obs_time in times:
                # Get the zero-based hour index in the daily file
                index = self.get_observation_hour_index(obs_time)

                # Extract the hourly slice corresponding to the observation end-time index
                slice_da = daily_dataset[config.obs_var_name].isel(time=index)

                # Accumulate the hourly slice into the total, initializing if this is the first slice
                accumulated_da = slice_da if accumulated_da is None else accumulated_da + slice_da

                # Increase the counter for each hourly slice accumulated
                count += 1

            # Close the daily dataset to free memory 
            daily_dataset.close()

        # Assert that at least one observation slice was accumulated to catch silent data gaps early.
        assert accumulated_da is not None, "No observation data for accumulation"

        # Log the number of hourly slices accumulated for debugging purposes.
        logger.debug("Accumulated %d obs hours from %s", count, valid_time)

        # Return the accumulated observation DataArray
        return accumulated_da


    def save_intermediate_precip(self: "FileManager",
                                 forecast_da: xr.DataArray,
                                 observation_da: xr.DataArray,
                                 cycle_start: datetime.datetime,
                                 valid_time: datetime.datetime,) -> None:
        """
        This method saves the intermediate forecast and observation precipitation fields for a specific valid time to a debug NetCDF file. The output file contains three variables: ``forecast``, ``observation``, and their difference ``difference``. The files are named using the valid-time string and written to a per-cycle precip subdirectory under the debug directory. This output is only produced when the ``save_intermediate`` flag is enabled in the configuration and can be used for debugging and verification of the accumulation and remapping processes. 

        Parameters:
            forecast_da (xr.DataArray): Regridded and masked forecast precipitation field.
            observation_da (xr.DataArray): Regridded and masked observation precipitation field.
            cycle_start (datetime.datetime): Initialisation time of the current forecast cycle.
            valid_time (datetime.datetime): Start of the current accumulation window.

        Returns:
            None
        """
        # Retrieve the configuration for use in path construction and file writing.
        config = self.config

        # Extract the init time string 
        init_str = cycle_start.strftime("%Y%m%d%H")

        # Compute the end time of the accumulation window
        end_time = valid_time + config.precip_accum

        # Extract the valid time string
        valid_str = end_time.strftime("%Y%m%d%H%M")

        # Specify the output directory for intermediate precip files 
        odir = os.path.join(
            config.resolve_relative_path(config.debug_dir), config.experiment_name, init_str, "precip"
        )

        # Ensure the output directory exists
        os.makedirs(odir, exist_ok=True)

        # Convert the valid time to seconds since the Unix epoch 
        time_val = (end_time - datetime.datetime(1970, 1, 1)).total_seconds()

        # Prepare the forecast DataArray for saving by expanding the time dimension 
        forecast_time_expanded = forecast_da.expand_dims({"time": [time_val]})

        # Prepare the observation DataArray for saving by expanding the time dimension 
        observation_time_expanded = observation_da.expand_dims({"time": [time_val]})

        # Create a dataset containing the forecast, observation, and their difference
        ds = xr.Dataset(
            {"forecast": forecast_time_expanded, "observation": observation_time_expanded, "difference": forecast_time_expanded - observation_time_expanded}
        )

        # Update time coordinate attributes to standard CF metadata for time variables
        ds["time"].attrs.update(
            standard_name="time",
            long_name="Time",
            units="seconds since 1970-01-01",
            calendar="standard",
        )

        # Update dataset attributes with experiment name, cycle start time, and valid time
        ds.attrs.update(
            experiment=config.experiment_name,
            cycle_start=cycle_start.strftime("%Y-%m-%d %H:%M:%S"),
            valid_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Specify zlib compression for all variables in the dataset 
        enc = {v: {"zlib": True, "complevel": config.compression_level} for v in ds.data_vars}

        # Specify the output file path for the intermediate precip NetCDF
        out = os.path.join(odir, f"precip_{valid_str}.nc")

        # Write the dataset to a NetCDF file with compression
        ds.to_netcdf(out, encoding=enc)

        # Log the successful write of the intermediate precip file for debugging purposes.
        logger.debug("Wrote intermediate precip: %s", out)


    def save_intermediate_binary(self: "FileManager",
                                 forecast_binary: xr.DataArray,
                                 observation_binary: xr.DataArray,
                                 cycle_start: datetime.datetime,
                                 valid_time: datetime.datetime,
                                 threshold: float,) -> None:
        """
        This method saves the intermediate binary exceedance masks for the forecast and observation fields for a specific valid time to a debug NetCDF file. The output file contains two variables: ``forecast_binary`` and ``observation_binary``, which are binary masks indicating exceedance of the specified threshold (values 0.0, 1.0, or NaN). The files are named using the valid-time string and threshold information, and written to a per-cycle binary subdirectory under the debug directory. This output is only produced when the ``save_intermediate`` flag is enabled in the configuration and can be used for debugging and verification of the thresholding process. 

        Parameters:
            forecast_binary (xr.DataArray): Binary exceedance mask for the forecast field (values 0.0, 1.0, or NaN).
            observation_binary (xr.DataArray): Binary exceedance mask for the observation field (values 0.0, 1.0, or NaN).
            cycle_start (datetime.datetime): Initialisation time of the current forecast cycle.
            valid_time (datetime.datetime): Start of the current accumulation window.
            threshold (float): Percentile threshold used to generate the binary masks.

        Returns:
            None
        """
        # Retrieve the configuration for use in path construction and file writing.
        config = self.config

        # Extract the init time string for filename formatting
        init_str = cycle_start.strftime("%Y%m%d%H")

        # Calculate the end time of the accumulation window for filename formatting
        end_time = valid_time + config.precip_accum

        # Format the valid time string for inclusion in the filename
        valid_str = end_time.strftime("%Y%m%d%H%M")

        # Specify the output directory for intermediate binary files
        odir = os.path.join(
            config.resolve_relative_path(config.debug_dir), config.experiment_name, init_str, "binary"
        )

        # Ensure the output directory exists for the intermediate binary files
        os.makedirs(odir, exist_ok=True)

        # Convert the valid time to seconds since the Unix epoch 
        time_val = (end_time - datetime.datetime(1970, 1, 1)).total_seconds()

        # Expand the forecast binary mask to include a time dimension with the same timestamp as the observation
        forecast_binary_expanded = forecast_binary.expand_dims({"time": [time_val]})

        # Expand the observation binary mask to include a time dimension with the same timestamp as the forecast 
        observation_binary_expanded = observation_binary.expand_dims({"time": [time_val]})

        # Create a dataset containing the forecast and observation binary masks
        ds = xr.Dataset({"forecast_binary": forecast_binary_expanded, "observation_binary": observation_binary_expanded})

        # Update time coordinate attributes to standard CF metadata for time variables
        ds["time"].attrs.update(
            standard_name="time",
            long_name="Time",
            units="seconds since 1970-01-01",
            calendar="standard",
        )

        # Update dataset attributes to include experiment name, cycle start time, valid time, and threshold percentile 
        ds.attrs.update(
            experiment=config.experiment_name,
            cycle_start=cycle_start.strftime("%Y-%m-%d %H:%M:%S"),
            valid_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            threshold_percentile=threshold,
        )

        # Apply zlib compression to all variables in the dataset to minimise disk usage
        enc = {v: {"zlib": True, "complevel": config.compression_level} for v in ds.data_vars}

        # Format the threshold value for inclusion in the filename
        tstr = format_threshold_for_filename(threshold)

        # Construct the output file path for the intermediate binary NetCDF
        out = os.path.join(odir, f"binary_thresh{tstr}_{valid_str}.nc")

        # Write the dataset containing the binary masks to a NetCDF file with compression
        ds.to_netcdf(out, encoding=enc)

        # Log the successful write of the intermediate binary file 
        logger.debug("Wrote intermediate binary: %s", out)


    def save_fss_results(self: "FileManager",
                         metrics_list: List[Dict[str, float]],
                         cycle_start: datetime.datetime,
                         region_name: str,
                         threshold: float,
                         window_size: int,) -> None:
        """
        This method persists FSS metrics for a single (cycle, region, threshold, window_size) combination. Each element of metrics_list is a dictionary containing ``fss``, ``pod``, ``far``, ``csi``, ``fbias``, and ``ets`` for one valid time. Files are written to the output directory hierarchy under ``<output_dir>/<experiment>/ExtendedFC/<init_string>/pp<step>h/`` with a standardised filename encoding the region name, threshold percentile, and window size. 

        Parameters:
            metrics_list (list of Dict[str, float]): One metrics dictionary per valid time, each containing keys ``fss``, ``pod``, ``far``, ``csi``, ``fbias``, ``ets``.
            cycle_start (datetime.datetime): Initialisation time of the forecast cycle.
            region_name (str): Verification domain name (e.g. ``"GLOBAL"``, ``"TROPICS"``).
            threshold (float): Percentile threshold used for the computation (e.g. 90.0).
            window_size (int): Neighbourhood window side-length in grid points.

        Returns:
            None
        """
        # Retrieve the configuration for use in path construction and file writing.
        config = self.config

        # Format the cycle initialization time as a string for inclusion in the output directory path 
        istr = cycle_start.strftime("%Y%m%d%H")

        # Use the effective precipitation accumulation hours for output directory and filename
        accum_h = config.effective_precip_accum_hours

        # Format the accumulation duration for inclusion in the output directory path (e.g. "pp3h" for 3-hour accum)
        pp_dir = f"pp{accum_h}h"

        # Construct the output directory path for the verification results 
        odir = os.path.join(
            config.resolve_relative_path(config.output_dir),
            config.experiment_name,
            "ExtendedFC",
            istr,
            pp_dir,
        )

        # Ensure the output directory exists 
        os.makedirs(odir, exist_ok=True)

        # Extract and format the threshold value for inclusion in the filename
        tstr = format_threshold_for_filename(threshold)

        # Construct a filename that encodes the region name, accumulation period, threshold percentile, and window size.
        fname = (
            f"modvx_metrics_type_neighborhood_{region_name.lower()}_"
            f"{accum_h}h_indep_thresh{tstr}percent_window{window_size}.nc"
        )

        # Construct the output file path
        out = os.path.join(odir, fname)

        # Define the list of metric keys to extract from the metrics dictionaries.
        metric_keys = _FSS_METRIC_KEYS

        # Initialize a dictionary to hold DataArrays for each metric
        data_vars = {}

        # Iterate over each metric key and construct a DataArray for that metric across all valid times
        for key in metric_keys:
            data_vars[key] = xr.DataArray(
                [m.get(key, float("nan")) for m in metrics_list],
                dims=["valid_time_index"],
            )

        # Create a dataset containing all metrics as separate variables
        ds = xr.Dataset(data_vars)

        # Write the dataset to a NetCDF file without compression since these files are already small 
        ds.to_netcdf(out)

        # Log the successful write of the FSS results file for debugging purposes.
        logger.info("Saved metrics → %s", out)


    def save_contingency_results(self: "FileManager",
                                 metrics_list: List[Dict[str, float]],
                                 cycle_start: datetime.datetime,
                                 region_name: str,
                                 threshold: float,) -> None:
        """
        This method persists contingency table metrics for a single (cycle, region, threshold) combination. Each element of metrics_list is a dictionary containing ``pod``, ``far``, ``csi``, ``fbias``, and ``ets`` for one valid time. Files are written to the output directory hierarchy under ``<output_dir>/<experiment>/ExtendedFC/<init_string>/pp<step>h/`` with a standardised filename encoding the region name and threshold percentile. 

        Parameters:
            metrics_list (list of Dict[str, float]): One metrics dictionary per valid time, each containing keys ``pod``, ``far``, ``csi``, ``fbias``, ``ets``.
            cycle_start (datetime.datetime): Initialisation time of the forecast cycle.
            region_name (str): Verification domain name (e.g. ``"GLOBAL"``, ``"TROPICS"``).
            threshold (float): Percentile threshold used for the computation (e.g. 90.0).

        Returns:
            None
        """
        # Retrieve the configuration for use in path construction and file writing.
        config = self.config

        # Format the cycle initialization time as a string for inclusion in the output directory path
        istr = cycle_start.strftime("%Y%m%d%H")

        # Use the effective precipitation accumulation hours for output directory and filename
        accum_h = config.effective_precip_accum_hours

        # Format the accumulation duration for inclusion in the output directory path (e.g. "pp3h" for 3-hour accum)
        pp_dir = f"pp{accum_h}h"

        # Construct the output directory path for the verification results
        odir = os.path.join(
            config.resolve_relative_path(config.output_dir),
            config.experiment_name,
            "ExtendedFC",
            istr,
            pp_dir,
        )

        # Ensure the output directory exists
        os.makedirs(odir, exist_ok=True)

        # Extract and format the threshold value for inclusion in the filename
        tstr = format_threshold_for_filename(threshold)

        # Construct a filename that encodes the region name, accumulation period, and threshold percentile for contingency results.
        fname = (
            f"modvx_metrics_type_contingency_{region_name.lower()}_"
            f"{accum_h}h_indep_thresh{tstr}percent.nc"
        )

        # Construct the output file path
        out = os.path.join(odir, fname)

        # Define the list of contingency metric keys to extract from the metrics dictionaries.
        metric_keys = _CONTINGENCY_KEYS

        # Initialize a dictionary to hold DataArrays for each metric
        data_vars = {}

        # Iterate over each metric key and construct a DataArray for that metric across all valid times
        for key in metric_keys:
            data_vars[key] = xr.DataArray(
                [m.get(key, float("nan")) for m in metrics_list],
                dims=["valid_time_index"],
            )

        # Create a dataset containing all metrics as separate variables
        ds = xr.Dataset(data_vars)

        # Write the dataset to a NetCDF file without compression since these files are already small
        ds.to_netcdf(out)

        # Log the successful write of the contingency results file for debugging purposes.
        logger.info("Saved contingency metrics → %s", out)


    @staticmethod
    def _extract_file_context(nc_file: str) -> Optional[tuple[str, str]]:
        """
        This static helper function attempts to extract the experiment name and initialization time from the file path of a FSS NetCDF file. It does this by splitting the file path into its components, identifying the segment following the "output" directory as the experiment name, and searching for a segment that matches the pattern of 10 digits (YYYYmmddHH) to identify the initialization time. If both pieces of information are successfully extracted, it returns them as a tuple. If the expected patterns are not found in the file path, it returns None to indicate that the context could not be parsed. 

        Parameters:
            nc_file (str): Full path to a FSS NetCDF file.

        Returns:
            Optional[Tuple[str, str]]: ``(experiment, init_time)`` when parseable, otherwise ``None``.
        """
        import re

        # Split the file path into its components to analyze the directory structure
        parts = Path(nc_file).parts

        # Find the index of the "output" directory in the path parts 
        try:
            out_idx = parts.index("output")
        except ValueError:
            return None

        # Extract the experiment name as the segment immediately following "output" in the path
        experiment = parts[out_idx + 1]

        # Extract the init_time by searching for a segment that matches the pattern of 10 digits (YYYYmmddHH)
        init_time = next((p for p in parts if re.match(r"^\d{10}$", p)), None)

        # If either the experiment name or init_time is not found, return None
        if init_time is None:
            return None
        
        # Return the extracted experiment name and init_time as a tuple
        return experiment, init_time


    @staticmethod
    def _parse_metric_values(ds: xr.Dataset, 
                             metric_keys: List[str],) -> tuple[int, Dict[str, Any]]:
        """
        This static helper function extracts metric values from an opened xarray Dataset based on a list of expected metric keys. It first determines the length of the records by checking the first available metric variable in the dataset, or falling back to any available variable if the expected keys are not present. Then, it iterates over the list of expected metric keys and attempts to extract their values from the dataset, filling in NaN for any missing keys. The result is a tuple containing the determined length of the records and a dictionary mapping each expected metric key to its corresponding array of values (or NaN if missing). This function provides a consistent way to extract metrics while handling cases where some expected variables may be absent from the dataset. 

        Parameters:
            ds (xarray.Dataset): Opened NetCDF dataset.
            metric_keys (list[str]): Metric variable names expected for this file type.

        Returns:
            Tuple[int, Dict[str, Any]]: ``(length, metric_vals)`` where length is the number of entries and metric_vals maps metric names to arrays.
        """
        # Initialize the length of the records to zero
        length = 0

        # Determine the length of the records by checking the first available metric variable in the dataset. 
        if metric_keys and metric_keys[0] in ds:
            length = len(ds[metric_keys[0]])
        elif "fss" in metric_keys and "__xarray_dataarray_variable__" in ds:
            length = len(ds["__xarray_dataarray_variable__"])
        elif ds.data_vars:
            first_key = list(ds.data_vars)[0]
            length = len(ds[first_key])

        # Initialize a dictionary to hold the metric values, filling in NaN for any missing keys
        metric_values: Dict[str, Any] = {}

        # Iterate over the expected metric keys and extract values from the dataset, or fill with NaN if the key is missing
        for key in metric_keys:
            if key in ds:
                metric_values[key] = ds[key].values
            elif key == "fss" and "__xarray_dataarray_variable__" in ds:
                metric_values[key] = ds["__xarray_dataarray_variable__"].values
            else:
                metric_values[key] = [float("nan")] * length

        # Return the determined length of the records and the dictionary of metric values
        return length, metric_values


    def _parse_records_from_nc_file(self: "FileManager", 
                                    nc_file: str) -> Optional[tuple[str, List[Dict]]]:
        """
        This internal helper method takes the path to a FSS NetCDF file, extracts the experiment name and initialization time from the file path, determines the type of metrics contained in the file (FSS or contingency), parses the metric values from the dataset, and constructs a list of metric record dictionaries containing all relevant information for each valid time index. Each record dictionary includes keys for initTime, leadTime, domain, thresh, window, and all metric values (fss, pod, far, csi, fbias, ets) with NaN for any missing metrics. The method returns a tuple of the experiment name and the list of metric records on success. If any step of the parsing process fails (e.g. unable to extract context from the file path, missing expected metadata patterns in the filename), it returns None to indicate that this file should be skipped. This function serves as a core component for reading and interpreting FSS NetCDF files into a structured format for further processing and CSV output. 

        Parameters:
            nc_file (str): Path to the FSS NetCDF file to parse.

        Returns:
            Optional[Tuple[str, List[Dict]]]: Tuple of ``(experiment, records)`` on success, or ``None`` when parsing fails.
        """
        from .utils import (
            extract_lead_time_hours_from_path,
            parse_fss_filename_metadata,
            parse_contingency_filename_metadata,
        )

        # Extract the experiment name and init_time from the file path using the provided helper function
        context = self._extract_file_context(nc_file)

        # Return None if the file context cannot be extracted
        if context is None:
            return None

        # Extract the experiment name and init_time from the context
        experiment, init_time = context

        # Extract the lead time in hours from the file path using the provided helper function
        lead_time = extract_lead_time_hours_from_path(nc_file)

        # Return None if the lead time cannot be extracted from the file path
        if lead_time is None:
            return None

        # Parse the filename to extract metadata such as domain, threshold, and window size.
        meta = parse_fss_filename_metadata(os.path.basename(nc_file))

        # Determine if this file is a contingency metrics file based on the presence of metadata patterns in the filename. 
        is_contingency = False

        # If the filename does not match the expected patterns for FSS metrics, attempt to parse it as a contingency metrics file.
        if meta is None:
            meta = parse_contingency_filename_metadata(os.path.basename(nc_file))
            is_contingency = meta is not None

        # Return None if the filename does not contain the expected metadata patterns
        if meta is None:
            return None

        # Open the NetCDF file using xarray 
        ds = xr.open_dataset(nc_file)

        # Extract metric values from the dataset based on file type
        if is_contingency:
            metric_keys = _CONTINGENCY_KEYS
        elif any(key in ds for key in _CONTINGENCY_KEYS):
            metric_keys = _ALL_METRIC_KEYS
        else:
            metric_keys = _FSS_METRIC_KEYS

        # Extract the length of the records and a dictionary of metric values from the dataset  
        length, metric_values = self._parse_metric_values(ds, metric_keys)

        # Build a full metric dictionary with NaN for missing keys
        metric_values_all: Dict[str, Any] = {
            k: metric_values.get(k, [float("nan")] * length)
            for k in _ALL_METRIC_KEYS
        }

        # Close the dataset to free resources 
        ds.close()

        # Extract the threshold value from metadata and convert it to a float for consistent typing in the records
        threshold_value = float(meta["thresh"])

        # Extract the window size from metadata if this is an FSS metrics file, otherwise set it to None
        window_value = None if is_contingency else int(meta["window"])

        # Construct a list of metric record dictionaries, one per valid time index, containing all relevant information for each record. 
        records = [
            {
                "initTime": init_time,
                "leadTime": lead_time * (index + 1),
                "domain": meta["domain"],
                "thresh": threshold_value,
                "window": window_value,
                **{k: metric_values_all[k][index] for k in _ALL_METRIC_KEYS},
            }
            for index in range(length)
        ]

        # Return the extracted experiment name and the list of metric records 
        return experiment, records


    def _write_experiment_csv(self: "FileManager", 
                              experiment: str, 
                              records: List[Dict], 
                              csv_dir: str,) -> None:
        """
        This internal helper method takes an experiment name, a list of metric record dictionaries, and a directory path, and writes the records to a CSV file named ``<experiment>.csv`` in the specified directory. It uses pandas to create a DataFrame from the list of dictionaries, sorts the records by initTime, leadTime, domain, thresh, and window for consistent ordering, and writes the DataFrame to a CSV file without the index column. After writing the file, it logs the successful write operation along with the number of records written for debugging purposes. This function serves as a core component for outputting the parsed metric records into a structured CSV format for each experiment. 

        Parameters:
            experiment (str): Experiment name used to form the CSV filename.
            records (List[Dict]): List of metric record dictionaries to persist.
            csv_dir (str): Directory to write the CSV file into.

        Returns:
            None
        """
        import pandas as pd

        # Create a DataFrame from the list of metric record dictionaries
        results_df = pd.DataFrame(records).sort_values(
            ["initTime", "leadTime", "domain", "thresh", "window"]
        )

        # Specify the full path to the CSV file for this experiment 
        csv_path = os.path.join(csv_dir, f"{experiment}.csv")

        # Write the DataFrame to a CSV file without the index column
        results_df.to_csv(csv_path, index=False)

        # Log the successful write of the CSV file with the number of records for debugging purposes.
        logger.info("Wrote %s (%d records)", csv_path, len(results_df))


    def extract_fss_to_csv(self: "FileManager",
                           output_dir: Optional[str] = None,
                           csv_dir: Optional[str] = None,) -> None:
        """
        This method orchestrates the extraction of FSS and contingency metrics from all relevant NetCDF files in the output directory hierarchy and writes them to CSV files for each experiment. It uses glob to recursively find all NetCDF files that match the expected pattern for FSS results, attempts to parse each file to extract the experiment name and metric records, aggregates the records by experiment, and then writes one CSV file per experiment containing all the parsed records. The method handles any exceptions during file parsing gracefully by logging a warning and skipping the problematic file, ensuring that one bad file does not disrupt the entire extraction process. This function serves as a convenient way to convert the structured NetCDF outputs of the verification process into a more accessible CSV format for analysis and reporting. 

        Parameters:
            output_dir (str, optional): Root output directory to scan; defaults to the configured output_dir.
            csv_dir (str, optional): Directory where CSV files will be written; defaults to the configured csv_dir.

        Returns:
            None
        """
        import glob

        # Retrieve the configuration for use in resolving default paths and other settings.
        config = self.config

        # If output_dir is not provided, resolve it from the configuration. 
        output_dir = output_dir or config.resolve_relative_path(config.output_dir)

        # If csv_dir is not provided, resolve it from the configuration. 
        csv_dir = csv_dir or config.resolve_relative_path(config.csv_dir)

        # Ensure the CSV output directory exists 
        os.makedirs(csv_dir, exist_ok=True)

        # Use glob to recursively find all NetCDF files in the output directory that match the pattern for results
        nc_files = glob.glob(
            os.path.join(output_dir, "**/ExtendedFC/**/*.nc"), recursive=True
        )

        # Log the number of metrics NetCDF files found for processing
        logger.info("Found %d metrics NetCDF files", len(nc_files))

        # Initialize a dictionary to hold lists of metric records for each experiment
        data: Dict[str, list] = defaultdict(list)

        # Loop through each discovered NetCDF file and attempt to parse metric records 
        for nc_file in nc_files:
            try:
                # Parse the NetCDF file to extract the experiment name and list of metric records.
                result = self._parse_records_from_nc_file(nc_file)

                # Extract the experiment name and records if parsing was successful, otherwise skip this file
                if result is not None:
                    # Unpack the experiment name and list of metric records 
                    experiment, records = result

                    # Aggregate the parsed records into the data dictionary under the corresponding experiment key
                    data[experiment].extend(records)
            except Exception:
                # Log any exceptions that occur during file parsing with the file path and stack trace for debugging purposes
                logger.warning("Error processing %s", nc_file, exc_info=True)

        # Write one CSV file per experiment with the aggregated records
        for experiment, records in data.items():
            self._write_experiment_csv(experiment, records, csv_dir)
