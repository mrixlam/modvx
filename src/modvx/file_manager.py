#!/usr/bin/env python3

"""
File I/O handling for MODvx.

This module defines the FileManager class, which is responsible for constructing file paths for forecasts and observations based on the provided configuration, loading data from NetCDF files into xarray structures, caching intermediate results to avoid redundant I/O, and saving output files with standardized naming conventions. The FileManager abstracts away the details of file organization and access patterns, allowing other components of the pipeline to focus on data processing and analysis without worrying about where and how the data is stored. By centralizing file management logic, we can ensure consistent handling of file paths, formats, and metadata across the entire verification workflow.

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
    """
    Centralise all filesystem operations for a modvx verification run. Responsibilities include locating forecast and observation files, loading and caching data with both in-memory and shared disk layers, writing intermediate debug fields as NetCDF, and persisting FSS results. The class is designed to be instantiated once per process and reused across all valid times in a cycle, allowing the in-memory observation cache to reduce redundant file reads.

    Parameters:
        config (ModvxConfig): Run configuration with directory paths, filename templates, and variable names.
    """


    def __init__(self, config: ModvxConfig) -> None:
        """
        This initialization method sets up the FileManager with the provided configuration, initializing in-memory caches for observations and forecasts, and ensuring that a shared disk cache directory is available for multiprocessing workers to share intermediate results. The configuration is stored as an instance attribute for use in path construction, loading, and caching operations throughout the verification workflow.

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


    def get_forecast_filepath(
        self,
        valid_time: datetime.datetime,
        init_string: str,
    ) -> str:
        """
        Construct the full filesystem path to a native MPAS diagnostic file. MPAS diag files are organised under the experiment directory in an ``ExtendedFC/<init_string>/`` subdirectory, named using the valid-time timestamp with second precision (e.g. ``diag.2024-09-17_01.00.00.nc``). The path is built from the configured forecast directory, experiment name, and the provided temporal identifiers without any filesystem access.

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


    def get_observation_filepath(self, date_key: str) -> str:
        """
        Resolve the path to a FIMERG daily observation NetCDF file for a given date. The function iterates through the configured vintage preference list (e.g. FNL, LTE) and returns the path to the first file that actually exists on disk. This allows graceful fallback when the preferred final-run data is not yet available. If no qualifying vintage is found, the first-preference path is returned and a FileNotFoundError will be raised at load time.

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
                f"E{date_key}T235959.{vintage}.V07B.SRCHHR.X3600Y1800.R0p1.FMT.nc"
            )

            # If the path exists, return it immediately
            if os.path.exists(path):
                return path
        
        # Return the full path to the observation file 
        return (
            f"{obs_dir}/IMERG.A01H.VLD{date_key}.S{date_key}T000000."
            f"E{date_key}T235959.{self.config.obs_vintage_preference[0]}.V07B.SRCHHR.X3600Y1800.R0p1.FMT.nc"
        )


    @staticmethod
    def get_observation_hour_index(time: datetime.datetime) -> int:
        """
        Map a FIMERG observation end-time to its zero-based hour index within the daily file. FIMERG daily files contain 24 hourly accumulation slices indexed 0–23, where index 0 corresponds to the hour ending at 01:00 UTC (covering 00:00–01:00) and index 23 corresponds to the hour ending at 00:00 UTC the following day (covering 23:00–00:00). This offset convention means that midnight (00:00) maps to index 23 rather than 0.

        Parameters:
            time (datetime.datetime): Observation end-time to convert to a file index.

        Returns:
            int: Zero-based time index (0–23) into the FIMERG daily NetCDF file.
        """
        # Return index 23 for midnight (00:00) and time.hour - 1 for all other hours 
        return 23 if time.hour == 0 else time.hour - 1


    @staticmethod
    def group_observation_times_by_date(
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        obs_interval: datetime.timedelta,
    ) -> Dict[str, List[datetime.datetime]]:
        """
        Group a sequence of hourly observation end-times by their corresponding daily FIMERG file. The function iterates from *start_time* to *end_time* inclusive at *obs_interval* steps, assigning each time to the appropriate date key. Times at midnight (00:00 UTC) are assigned to the previous calendar day's file, consistent with FIMERG's end-of-hour timestamp convention. The result allows loading each daily file exactly once when accumulating observations over a multi-hour window.

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


    def load_region_mask(self, mask_filepath: str) -> Tuple[xr.DataArray, str]:
        """
        Load a binary verification-domain mask from a NetCDF file. After opening, dimension names are standardised to ``latitude``/``longitude`` and longitudes are converted to the [0, 360] convention for consistency with observation data. The first non-coordinate variable in the file is treated as the mask variable. Raises descriptive errors when the file is missing or does not contain a suitable mask variable.

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


    def _forecast_cache_key(
        self, init_string: str, valid_time: datetime.datetime,
    ) -> str:
        """
        Generate a deterministic cache key for a forecast accumulation field that is unique per experiment. The key encodes the experiment name, cycle initialisation string, valid time, and forecast step duration so that different experiments running over the same verification period never share a cache entry. Without the experiment name, a cache written by one experiment could be silently reused by a subsequent experiment with the same cycle and valid-time range, producing identical (incorrect) scores for every experiment after the first.

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


    def _load_cache_entry(
        self, key: str, in_memory_cache: Dict[str, xr.DataArray],
    ) -> Optional[xr.DataArray]:
        """
        This helper checks both the in-memory cache dict and the shared disk cache directory for a cached DataArray corresponding to the provided key. It first looks up the key in the in-memory dict, returning the cached DataArray if found. If not present in memory, it checks if a NetCDF file with the key name exists in the shared disk cache directory. If such a file exists, it loads the DataArray from disk, stores it in the in-memory cache for future quick access, and returns it. If the key is not found in either cache layer, the method returns None, indicating that the caller needs to compute and cache the value.

        Parameters:
            key (str): Cache key string to lookup.
            mem_cache (Dict[str, xarray.DataArray]): In-memory cache mapping.

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


    def _save_cache_entry(
        self, key: str, data_array: xr.DataArray, in_memory_cache: Dict[str, xr.DataArray],
    ) -> None:
        """
        This helper saves a DataArray to both the in-memory cache dict and the shared disk cache directory under a filename derived from the provided key. It first stores the DataArray in the in-memory dict for fast access within the current process. If a shared disk cache directory is configured, it then writes the DataArray to a NetCDF file named ``{key}.nc`` within that directory. The method ensures that the cache directory exists and uses an atomic file write (write to a temp file and rename

        Parameters:
            key (str): Cache key under which to store the DataArray.
            da (xarray.DataArray): DataArray to persist.
            mem_cache (Dict[str, xarray.DataArray]): In-memory cache mapping.

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


    def _compute_forecast_accumulation(
        self, valid_time: datetime.datetime, init_string: str,
    ) -> xr.DataArray:
        """
        This internal method performs the actual computation of accumulated precipitation from MPAS diag files without any caching. It loads the cumulative precipitation at the start and end of the accumulation window, computes the difference to get the accumulation over the forecast step, and then remaps the result from the native MPAS mesh to a regular lat-lon grid using mpasdiag's remapping utilities. The method assumes that the input valid_time and init_string are valid and that the corresponding MPAS diag files exist; it does not handle any exceptions related to missing files or data.

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


    def accumulate_forecasts(
        self,
        valid_time: datetime.datetime,
        init_string: str,
    ) -> xr.DataArray:
        """
        Compute accumulated precipitation for one forecast step from native MPAS diag files. MPAS diag files store cumulative ``rainc`` and ``rainnc`` from model initialisation, so the accumulation for a window is computed by subtracting the value at the window start from the value at the window end. The result is then remapped from the native unstructured MPAS mesh to a regular lat-lon grid at the configured resolution using the ``mpasdiag`` library.

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


    def _observation_cache_key(self, valid_time: datetime.datetime) -> str:
        """
        Generate a deterministic string cache key for an accumulated observation field. The key encodes the valid time and forecast step duration, ensuring that cache entries are unique per (valid_time, accumulation_length) combination. The same key is used to check both the in-memory dict cache and the shared disk cache directory, making it straightforward to coordinate cache reads across workers.

        Parameters:
            valid_time (datetime.datetime): Start of the observation accumulation window.

        Returns:
            str: Cache key string in the format ``"obs_accum_<YYYYmmddHH>_<N>h"``.
        """
        # Calculate the forecast step in hours from the configuration's forecast_step timedelta
        step_h = int(self.config.forecast_step.total_seconds() / 3600)

        # Return a formatted cache key string for the observation accumulation 
        return f"obs_accum_{valid_time.strftime('%Y%m%d%H')}_{step_h}h"


    def accumulate_observations(
        self,
        valid_time: datetime.datetime,
    ) -> xr.DataArray:
        """
        Load and accumulate FIMERG observation data with multi-level caching. Three cache layers are checked in order: (1) an in-memory dict per process, which avoids re-loading data when the same valid time appears in multiple cycles; (2) a shared disk cache directory containing pre-computed NetCDF files, which allows multiprocessing workers to share work; and (3) the original FIMERG daily source files as the final fallback. Successfully loaded data is stored in both cache layers for future use.

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


    def _compute_observation_accumulation_raw(
        self,
        valid_time: datetime.datetime,
    ) -> xr.DataArray:
        """
        Load and sum hourly FIMERG observation slices over one forecast step from source files. This internal method performs the actual file I/O without any caching, making it the fallback called by accumulate_observations when no cached result is available. Hourly slices are grouped by daily file to minimise the number of NetCDF files opened. The method asserts that at least one observation slice was accumulated to catch silent data gaps early.

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


    def save_intermediate_precip(
        self,
        forecast_da: xr.DataArray,
        observation_da: xr.DataArray,
        cycle_start: datetime.datetime,
        valid_time: datetime.datetime,
    ) -> None:
        """
        Save regridded and masked forecast and observation precipitation fields to a debug NetCDF. The output file contains three variables: ``forecast``, ``observation``, and ``difference``, each with a time dimension encoded as seconds since the Unix epoch. Files are written to a per-cycle subdirectory under the configured debug directory and are only produced when the ``save_intermediate`` flag is enabled. zlib compression is applied at the configured level to minimise disk usage.

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
        end_time = valid_time + config.forecast_step

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


    def save_intermediate_binary(
        self,
        forecast_binary: xr.DataArray,
        observation_binary: xr.DataArray,
        cycle_start: datetime.datetime,
        valid_time: datetime.datetime,
        threshold: float,
    ) -> None:
        """
        Save binary exceedance mask fields to a debug NetCDF file for a specific threshold. The output file contains two variables, ``forecast_binary`` and ``observation_binary``, representing the 0/1/NaN exceedance masks produced by PerfMetrics.generate_binary_mask. Files are named using the threshold value and valid-time string and written to a per-cycle binary subdirectory under the debug directory. This output is only produced when the ``save_intermediate`` flag is enabled in the configuration.

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
        end_time = valid_time + config.forecast_step

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


    def save_fss_results(
        self,
        metrics_list: List[Dict[str, float]],
        cycle_start: datetime.datetime,
        region_name: str,
        threshold: float,
        window_size: int,
    ) -> None:
        """
        Persist FSS metrics for a single (cycle, region, threshold, window) combination. Each element of metrics_list is a dictionary containing ``fss`` for one valid time. Files are written to the output directory hierarchy under ``<output_dir>/<experiment>/ExtendedFC/<init_string>/pp<step>h/`` with a standardised filename encoding the region name, threshold percentile, and window size. All metric arrays are stored as variables inside a single NetCDF dataset so that extract_fss_to_csv can reconstruct the full record.

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

        # Calculate the forecast step in hours from the configuration's forecast_step timedelta 
        step_h = int(config.forecast_step.total_seconds() / 3600)

        # Format the forecast step duration for inclusion in the output directory path (e.g. "pp24h" for a 24-hour step)
        pp_dir = f"pp{step_h}h"

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

        # Construct a filename that encodes the region name, forecast step, threshold percentile, and window size.
        fname = (
            f"modvx_metrics_type_neighborhood_{region_name.lower()}_"
            f"{step_h}h_indep_thresh{tstr}percent_window{window_size}.nc"
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


    def save_contingency_results(
        self,
        metrics_list: List[Dict[str, float]],
        cycle_start: datetime.datetime,
        region_name: str,
        threshold: float,
    ) -> None:
        """
        Persist contingency metrics for a single (cycle, region, threshold) combination. Each element of metrics_list is a dictionary containing ``pod``, ``far``, ``csi``, ``fbias``, and ``ets`` for one valid time. Files are written to the output directory hierarchy under ``<output_dir>/<experiment>/ExtendedFC/<init_string>/pp<step>h/`` with a standardised filename encoding the region name and threshold percentile (no window size).

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

        # Calculate the forecast step in hours from the configuration's forecast_step timedelta
        step_h = int(config.forecast_step.total_seconds() / 3600)

        # Format the forecast step duration for inclusion in the output directory path (e.g. "pp24h" for a 24-hour step)
        pp_dir = f"pp{step_h}h"

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

        # Construct a filename that encodes the region name and threshold percentile for contingency results.
        fname = (
            f"modvx_metrics_type_contingency_{region_name.lower()}_"
            f"{step_h}h_indep_thresh{tstr}percent.nc"
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
        This helper function extracts the experiment name and init_time from a given FSS NetCDF file path. It looks for the "output" directory in the path, then assumes the next segment is the experiment name. It also searches for a segment matching a 10-digit pattern (YYYYmmddHH) to identify the init_time. If both pieces of information are successfully extracted, it returns them as a tuple; otherwise, it returns None to indicate that the file does not conform to the expected structure.

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
    def _parse_metric_values(
        ds: xr.Dataset, metric_keys: List[str],
    ) -> tuple[int, Dict[str, Any]]:
        """
        This helper function extracts metric values from an xarray Dataset based on a list of expected metric keys. It first determines the length of the records by checking the length of the first available metric variable in the dataset. Then, for each expected metric key, it attempts to extract the corresponding values from the dataset. If a key is missing, it fills in an array of NaN values with the same length as the records. The function returns both the determined length and a dictionary mapping each metric key to its extracted values or NaN arrays.

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


    def _parse_records_from_nc_file(self, nc_file: str) -> Optional[tuple[str, List[Dict]]]:
        """
        This method parses a single FSS NetCDF file to extract the experiment name, init_time, lead_time, and metric values for each valid time index. It uses helper functions to extract context from the filename and read metric values from the dataset. The result is a list of dictionaries, each representing a record with keys for initTime, leadTime, domain, thresh, window, and all metrics. If any step of the parsing fails (e.g. missing expected variables or filename patterns), the method returns None to indicate that the file could not be processed.

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


    def _write_experiment_csv(
        self, experiment: str, records: List[Dict], csv_dir: str,
    ) -> None:
        """
        This method takes a list of metric record dictionaries for a single experiment and writes them to a CSV file in the specified directory. It uses pandas to create a DataFrame from the records, sorts it by initTime, leadTime, domain, thresh, and window for consistent ordering, and then writes it to a CSV file named ``<experiment>.csv``. The method logs the number of records written and the path to the output file.

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


    def extract_fss_to_csv(
        self,
        output_dir: Optional[str] = None,
        csv_dir: Optional[str] = None,
    ) -> None:
        """
        Scan the output directory tree for metrics NetCDF files and write one CSV per experiment. All NetCDF files matching the ``**/ExtendedFC/**/*.nc`` glob pattern are discovered, and their filenames are parsed to extract domain, threshold, and window metadata. Lead times are extracted from the directory path via extract_lead_time_hours. Results are aggregated into a pandas DataFrame per experiment and written to ``<csv_dir>/<experiment>.csv``.

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
