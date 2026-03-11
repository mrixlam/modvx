#!/usr/bin/env python3

"""
File I/O for modvx.

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

_METRIC_KEYS: list[str] = ["fss", "pod", "far", "csi", "fbias", "ets"]


class FileManager:
    """
    Centralise all filesystem operations for a modvx verification run. Responsibilities include locating forecast and observation files, loading and caching data with both in-memory and shared disk layers, writing intermediate debug fields as NetCDF, and persisting FSS results. The class is designed to be instantiated once per process and reused across all valid times in a cycle, allowing the in-memory observation cache to reduce redundant file reads.

    Parameters:
        config (ModvxConfig): Run configuration with directory paths, filename templates,
            and variable names.
    """

    def __init__(self, config: ModvxConfig) -> None:
        self.config = config
        self._obs_mem_cache: Dict[str, xr.DataArray] = {}
        self._fcst_mem_cache: Dict[str, xr.DataArray] = {}

        # Ensure a shared disk cache directory is always available so that
        # forecast and observation caches work across MPI ranks / workers.
        if self.config.cache_dir is None:
            self.config.cache_dir = os.path.join(
                config.resolve_relative_path(config.output_dir), ".obs_cache",
            )

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    def get_forecast_filepath(
        self,
        valid_time: datetime.datetime,
        init_string: str,
    ) -> str:
        """
        Construct the full filesystem path to a native MPAS diagnostic file. MPAS diag files are organised under the experiment directory in an ``ExtendedFC/<init_string>/`` subdirectory, named using the valid-time timestamp with second precision (e.g. ``diag.2024-09-17_01.00.00.nc``). The path is built from the configured forecast directory, experiment name, and the provided temporal identifiers without any filesystem access.

        Parameters:
            valid_time (datetime.datetime): Forecast valid time, used to format the filename.
            init_string (str): Cycle initialisation string in ``YYYYmmddHH`` format
                (e.g. ``"2024091700"``).

        Returns:
            str: Full path to the MPAS diag NetCDF file for the specified valid time and cycle.
        """
        config = self.config
        timestamp_str = valid_time.strftime("%Y-%m-%d_%H.%M.%S")
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
        obs_dir = self.config.resolve_relative_path(self.config.obs_dir)
        for vintage in self.config.obs_vintage_preference:
            path = (
                f"{obs_dir}/IMERG.A01H.VLD{date_key}.S{date_key}T000000."
                f"E{date_key}T235959.{vintage}.V07B.SRCHHR.X3600Y1800.R0p1.FMT.nc"
            )
            if os.path.exists(path):
                return path
        # Fallback — let caller handle FileNotFoundError
        return (
            f"{obs_dir}/IMERG.A01H.VLD{date_key}.S{date_key}T000000."
            f"E{date_key}T235959.{self.config.obs_vintage_preference[0]}.V07B.SRCHHR.X3600Y1800.R0p1.FMT.nc"
        )

    # ------------------------------------------------------------------
    # FIMERG time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_observation_hour_index(time: datetime.datetime) -> int:
        """
        Map a FIMERG observation end-time to its zero-based hour index within the daily file. FIMERG daily files contain 24 hourly accumulation slices indexed 0–23, where index 0 corresponds to the hour ending at 01:00 UTC (covering 00:00–01:00) and index 23 corresponds to the hour ending at 00:00 UTC the following day (covering 23:00–00:00). This offset convention means that midnight (00:00) maps to index 23 rather than 0.

        Parameters:
            time (datetime.datetime): Observation end-time to convert to a file index.

        Returns:
            int: Zero-based time index (0–23) into the FIMERG daily NetCDF file.
        """
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
            Dict[str, List[datetime.datetime]]: Mapping of ``YYYYmmdd`` date keys to lists
                of datetime objects belonging to that file.
        """
        times_by_date: Dict[str, List[datetime.datetime]] = defaultdict(list)
        current = start_time
        while current <= end_time:
            if current.hour == 0:
                date_key = (current - datetime.timedelta(days=1)).strftime("%Y%m%d")
            else:
                date_key = current.strftime("%Y%m%d")
            times_by_date[date_key].append(current)
            current += obs_interval
        return times_by_date

    # ------------------------------------------------------------------
    # Mask loading
    # ------------------------------------------------------------------

    def load_region_mask(self, mask_filepath: str) -> Tuple[xr.DataArray, str]:
        """
        Load a binary verification-domain mask from a NetCDF file. After opening, dimension names are standardised to ``latitude``/``longitude`` and longitudes are converted to the [0, 360] convention for consistency with observation data. The first non-coordinate variable in the file is treated as the mask variable. Raises descriptive errors when the file is missing or does not contain a suitable mask variable.

        Parameters:
            mask_filepath (str): Full path to the mask NetCDF file.

        Returns:
            Tuple[xr.DataArray, str]: Tuple of ``(mask_array, variable_name)`` where the
                mask has standard coordinate names and [0, 360] longitudes.
        """
        if not os.path.exists(mask_filepath):
            raise FileNotFoundError(f"Mask file not found: {mask_filepath}")

        ds = xr.open_dataset(mask_filepath)
        data_vars = [
            v for v in ds.data_vars
            if v not in {"lat", "lon", "latitude", "longitude"}
        ]
        if not data_vars:
            raise ValueError(f"No mask variable found in {mask_filepath}")

        var_name = str(data_vars[0])
        mask = standardize_coords(ds[var_name])
        mask = normalize_longitude(mask, "0_360")

        logger.debug(
            "Loaded mask '%s' from %s — %d/%d valid (%.1f%%)",
            var_name,
            mask_filepath,
            (mask.values > 0).sum(),
            mask.size,
            100 * (mask.values > 0).sum() / mask.size,
        )
        return mask, var_name

    # ------------------------------------------------------------------
    # Forecast accumulation (with shared cache)
    # ------------------------------------------------------------------

    def _forecast_cache_key(
        self, init_string: str, valid_time: datetime.datetime,
    ) -> str:
        """
        This method generates a deterministic cache key string for a forecast accumulation field based on the cycle initialisation time, valid time, and forecast step duration. The key format encodes the init_string, valid_time (formatted as YYYYmmddHH), and the forecast step in hours (e.g. "fcst_accum_2024091700_202409171200_6h"). This consistent key generation allows both the in-memory and shared disk caches to be accessed using the same identifier, ensuring that once a forecast accumulation is computed by any worker it can be reused across all workers for that valid time and cycle without redundant computation or file I/O.

        Parameters:
            init_string (str): Cycle initialisation string in ``YYYYmmddHH`` format.
            valid_time (datetime.datetime): Start of the accumulation window.

        Returns:
            str: Deterministic cache key string.
        """
        step_h = int(self.config.forecast_step.total_seconds() / 3600)
        return f"fcst_accum_{init_string}_{valid_time.strftime('%Y%m%d%H')}_{step_h}h"

    # --- shared cache helpers ---

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
        if key in in_memory_cache:
            logger.debug("Cache hit (memory): %s", key)
            return in_memory_cache[key]
        cache_dir = self.config.cache_dir
        if cache_dir:
            disk_path = os.path.join(cache_dir, f"{key}.nc")
            if os.path.exists(disk_path):
                logger.debug("Cache hit (disk): %s", disk_path)
                dataset = xr.load_dataset(disk_path)
                data_array = dataset[list(dataset.data_vars)[0]]
                in_memory_cache[key] = data_array
                return data_array
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
        in_memory_cache[key] = data_array
        cache_dir = self.config.cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            disk_path = os.path.join(cache_dir, f"{key}.nc")
            tmp_path = f"{disk_path}.tmp.{os.getpid()}"
            data_array.to_netcdf(tmp_path)
            os.rename(tmp_path, disk_path)  # atomic on POSIX
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

        config = self.config
        grid_file = config.resolve_relative_path(config.mpas_grid_file)
        window_end = valid_time + config.forecast_step

        end_file = self.get_forecast_filepath(window_end, init_string)
        start_file = self.get_forecast_filepath(valid_time, init_string)
        logger.debug("MPAS accum: %s → %s", start_file, end_file)

        end_precip = load_mpas_precip(end_file, grid_file)
        start_precip = load_mpas_precip(start_file, grid_file)
        accum_mesh = end_precip - start_precip
        del end_precip, start_precip

        accum_mesh.attrs["units"] = "mm"
        accum_mesh.attrs["long_name"] = f"{int(config.forecast_step_hours)}h accumulated precipitation"

        remapped_da = remap_to_latlon(accum_mesh, end_file, grid_file, config.mpas_remap_resolution)
        del accum_mesh

        logger.debug(
            "MPAS accumulated %dh precip from %s, remapped to %.3f° grid",
            config.forecast_step_hours, valid_time, config.mpas_remap_resolution,
        )
        return remapped_da

    def accumulate_forecasts(
        self,
        valid_time: datetime.datetime,
        init_string: str,
    ) -> xr.DataArray:
        """
        Compute accumulated precipitation for one forecast step from native MPAS diag files. MPAS diag files store cumulative ``rainc`` and ``rainnc`` from model initialisation, so the accumulation for a window is computed by subtracting the value at the window start from the value at the window end. The result is then remapped from the native unstructured MPAS mesh to a regular lat-lon grid at the configured resolution using the ``mpasdiag`` library.

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window
                (window end = valid_time + forecast_step).
            init_string (str): Cycle initialisation string in ``YYYYmmddHH`` format
                (e.g. ``"2024091700"``).

        Returns:
            xr.DataArray: Accumulated precipitation on a regular lat-lon grid in millimetres.
        """
        key = self._forecast_cache_key(init_string, valid_time)
        cached = self._load_cache_entry(key, self._fcst_mem_cache)
        if cached is not None:
            return cached
        remapped = self._compute_forecast_accumulation(valid_time, init_string)
        self._save_cache_entry(key, remapped, self._fcst_mem_cache)
        return remapped

    # ------------------------------------------------------------------
    # Observation accumulation (with shared cache)
    # ------------------------------------------------------------------

    def _observation_cache_key(self, valid_time: datetime.datetime) -> str:
        """
        Generate a deterministic string cache key for an accumulated observation field. The key encodes the valid time and forecast step duration, ensuring that cache entries are unique per (valid_time, accumulation_length) combination. The same key is used to check both the in-memory dict cache and the shared disk cache directory, making it straightforward to coordinate cache reads across workers.

        Parameters:
            valid_time (datetime.datetime): Start of the observation accumulation window.

        Returns:
            str: Cache key string in the format ``"obs_accum_<YYYYmmddHH>_<N>h"``.
        """
        step_h = int(self.config.forecast_step.total_seconds() / 3600)
        return f"obs_accum_{valid_time.strftime('%Y%m%d%H')}_{step_h}h"

    def accumulate_observations(
        self,
        valid_time: datetime.datetime,
    ) -> xr.DataArray:
        """
        Load and accumulate FIMERG observation data with multi-level caching. Three cache layers are checked in order: (1) an in-memory dict per process, which avoids re-loading data when the same valid time appears in multiple cycles; (2) a shared disk cache directory containing pre-computed NetCDF files, which allows multiprocessing workers to share work; and (3) the original FIMERG daily source files as the final fallback. Successfully loaded data is stored in both cache layers for future use.

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window, aligned with
                the forecast valid time.

        Returns:
            xr.DataArray: Accumulated hourly precipitation sum over the forecast step period.
        """
        key = self._observation_cache_key(valid_time)
        cached = self._load_cache_entry(key, self._obs_mem_cache)
        if cached is not None:
            return cached
        accumulated_da = self._compute_observation_accumulation_raw(valid_time)
        self._save_cache_entry(key, accumulated_da, self._obs_mem_cache)
        return accumulated_da

    def _compute_observation_accumulation_raw(
        self,
        valid_time: datetime.datetime,
    ) -> xr.DataArray:
        """
        Load and sum hourly FIMERG observation slices over one forecast step from source files. This internal method performs the actual file I/O without any caching, making it the fallback called by accumulate_observations when no cached result is available. Hourly slices are grouped by daily file to minimise the number of NetCDF files opened. The method asserts that at least one observation slice was accumulated to catch silent data gaps early.

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window
                (window end = valid_time + forecast_step).

        Returns:
            xr.DataArray: Accumulated hourly precipitation sum over the forecast step period.
        """
        config = self.config
        start_time = valid_time + config.observation_interval
        end_time = valid_time + config.forecast_step

        times_by_date = self.group_observation_times_by_date(
            start_time, end_time, config.observation_interval
        )

        accumulated_da: Optional[xr.DataArray] = None
        count = 0

        for date_key, times in sorted(times_by_date.items()):
            daily_file = self.get_observation_filepath(date_key)
            logger.debug("Loading obs file: %s", daily_file)
            daily_dataset = xr.load_dataset(daily_file)

            for obs_time in times:
                index = self.get_observation_hour_index(obs_time)
                slice_da = daily_dataset[config.obs_var_name].isel(time=index)
                accumulated_da = slice_da if accumulated_da is None else accumulated_da + slice_da
                count += 1

            daily_dataset.close()

        assert accumulated_da is not None, "No observation data for accumulation"
        logger.debug("Accumulated %d obs hours from %s", count, valid_time)
        return accumulated_da

    # ------------------------------------------------------------------
    # Intermediate field saving
    # ------------------------------------------------------------------

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
        config = self.config
        init_str = cycle_start.strftime("%Y%m%d%H")
        end_time = valid_time + config.forecast_step
        valid_str = end_time.strftime("%Y%m%d%H%M")

        odir = os.path.join(
            config.resolve_path(config.debug_dir), config.experiment_name, init_str, "precip"
        )
        os.makedirs(odir, exist_ok=True)

        time_val = (end_time - datetime.datetime(1970, 1, 1)).total_seconds()
        forecast_time_expanded = forecast_da.expand_dims({"time": [time_val]})
        observation_time_expanded = observation_da.expand_dims({"time": [time_val]})

        ds = xr.Dataset(
            {"forecast": forecast_time_expanded, "observation": observation_time_expanded, "difference": forecast_time_expanded - observation_time_expanded}
        )
        ds["time"].attrs.update(
            standard_name="time",
            long_name="Time",
            units="seconds since 1970-01-01",
            calendar="standard",
        )
        ds.attrs.update(
            experiment=config.experiment_name,
            cycle_start=cycle_start.strftime("%Y-%m-%d %H:%M:%S"),
            valid_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        enc = {v: {"zlib": True, "complevel": config.compression_level} for v in ds.data_vars}
        out = os.path.join(odir, f"precip_{valid_str}.nc")
        ds.to_netcdf(out, encoding=enc)
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
            forecast_binary (xr.DataArray): Binary exceedance mask for the forecast field
                (values 0.0, 1.0, or NaN).
            observation_binary (xr.DataArray): Binary exceedance mask for the observation
                field (values 0.0, 1.0, or NaN).
            cycle_start (datetime.datetime): Initialisation time of the current forecast cycle.
            valid_time (datetime.datetime): Start of the current accumulation window.
            threshold (float): Percentile threshold used to generate the binary masks.

        Returns:
            None
        """
        config = self.config
        init_str = cycle_start.strftime("%Y%m%d%H")
        end_time = valid_time + config.forecast_step
        valid_str = end_time.strftime("%Y%m%d%H%M")

        odir = os.path.join(
            config.resolve_path(config.debug_dir), config.experiment_name, init_str, "binary"
        )
        os.makedirs(odir, exist_ok=True)

        time_val = (end_time - datetime.datetime(1970, 1, 1)).total_seconds()
        forecast_binary_expanded = forecast_binary.expand_dims({"time": [time_val]})
        observation_binary_expanded = observation_binary.expand_dims({"time": [time_val]})

        ds = xr.Dataset({"forecast_binary": forecast_binary_expanded, "observation_binary": observation_binary_expanded})
        ds["time"].attrs.update(
            standard_name="time",
            long_name="Time",
            units="seconds since 1970-01-01",
            calendar="standard",
        )
        ds.attrs.update(
            experiment=config.experiment_name,
            cycle_start=cycle_start.strftime("%Y-%m-%d %H:%M:%S"),
            valid_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            threshold_percentile=threshold,
        )

        enc = {v: {"zlib": True, "complevel": config.compression_level} for v in ds.data_vars}
        tstr = format_threshold_for_filename(threshold)
        out = os.path.join(odir, f"binary_thresh{tstr}_{valid_str}.nc")
        ds.to_netcdf(out, encoding=enc)
        logger.debug("Wrote intermediate binary: %s", out)

    # ------------------------------------------------------------------
    # FSS result saving
    # ------------------------------------------------------------------

    def save_fss_results(
        self,
        metrics_list: List[Dict[str, float]],
        cycle_start: datetime.datetime,
        region_name: str,
        threshold: float,
        window_size: int,
    ) -> None:
        """
        Persist verification metrics for a single (cycle, region, threshold, window) combination. Each element of metrics_list is a dictionary of metric values (fss, pod, far, csi, fbias, ets) for one valid time. Files are written to the output directory hierarchy under ``<output_dir>/<experiment>/ExtendedFC/<init_string>/pp<step>h/`` with a standardised filename encoding the region name, threshold percentile, and window size. All metric arrays are stored as variables inside a single NetCDF dataset so that extract_fss_to_csv can reconstruct the full record.

        Parameters:
            metrics_list (list of Dict[str, float]): One metrics dictionary per valid time,
                each containing keys ``fss``, ``pod``, ``far``, ``csi``, ``fbias``, ``ets``.
            cycle_start (datetime.datetime): Initialisation time of the forecast cycle.
            region_name (str): Verification domain name (e.g. ``"GLOBAL"``, ``"TROPICS"``).
            threshold (float): Percentile threshold used for the computation (e.g. 90.0).
            window_size (int): Neighbourhood window side-length in grid points.

        Returns:
            None
        """
        config = self.config
        istr = cycle_start.strftime("%Y%m%d%H")
        step_h = int(config.forecast_step.total_seconds() / 3600)
        pp_dir = f"pp{step_h}h"

        odir = os.path.join(
            config.resolve_path(config.output_dir),
            config.experiment_name,
            "ExtendedFC",
            istr,
            pp_dir,
        )
        os.makedirs(odir, exist_ok=True)

        tstr = format_threshold_for_filename(threshold)
        fname = f"{region_name}_FSS_{step_h}h_indep_thresh{tstr}percent_window{window_size}.nc"
        out = os.path.join(odir, fname)

        metric_keys = ["fss", "pod", "far", "csi", "fbias", "ets"]
        data_vars = {}
        for key in metric_keys:
            data_vars[key] = xr.DataArray(
                [m.get(key, float("nan")) for m in metrics_list],
                dims=["valid_time_index"],
            )

        ds = xr.Dataset(data_vars)
        ds.to_netcdf(out)
        logger.info("Saved metrics → %s", out)

    # ------------------------------------------------------------------
    # CSV extraction (from s3)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_file_context(nc_file: str) -> Optional[tuple[str, str]]:
        """
        This helper function extracts the experiment name and init_time from a given FSS NetCDF file path. It looks for the "output" directory in the path, then assumes the next segment is the experiment name. It also searches for a segment matching a 10-digit pattern (YYYYmmddHH) to identify the init_time. If both pieces of information are successfully extracted, it returns them as a tuple; otherwise, it returns None to indicate that the file does not conform to the expected structure.

        Parameters:
            nc_file (str): Full path to a FSS NetCDF file.

        Returns:
            Optional[Tuple[str, str]]: ``(experiment, init_time)`` when parseable,
            otherwise ``None``.
        """
        import re
        parts = Path(nc_file).parts
        try:
            out_idx = parts.index("output")
        except ValueError:
            return None
        experiment = parts[out_idx + 1]
        init_time = next((p for p in parts if re.match(r"^\d{10}$", p)), None)
        if init_time is None:
            return None
        return experiment, init_time

    @staticmethod
    def _parse_metric_values(ds: xr.Dataset) -> tuple[Any, Dict[str, Any]]:
        """
        This helper function reads metric values from a FSS NetCDF dataset. It first checks if the dataset contains a variable named "fss". If it does, it assumes that all metrics are present as separate variables and reads them directly. If "fss" is not found, it falls back to reading the single unnamed variable (which would be the case if the file was written with a single DataArray without explicit variable names) and treats it as the FSS values, while filling all other metrics with NaN. The function returns a tuple of the FSS values array and a dictionary mapping each metric name to its corresponding array.

        Parameters:
            ds (xarray.Dataset): Opened FSS dataset.

        Returns:
            Tuple[Any, Dict[str, Any]]: ``(fss_vals, metric_vals)`` where fss_vals
                is a 1-D array of FSS values and metric_vals maps metric names to arrays.
        """
        if "fss" in ds:
            fss_values = ds["fss"].values
            metric_values: Dict[str, Any] = {
                k: ds[k].values if k in ds else [float("nan")] * len(fss_values)
                for k in _METRIC_KEYS
            }
        else:
            fss_values = ds["__xarray_dataarray_variable__"].values
            metric_values = {
                "fss": fss_values,
                **{k: [float("nan")] * len(fss_values) for k in _METRIC_KEYS if k != "fss"},
            }
        return fss_values, metric_values

    def _parse_records_from_nc_file(self, nc_file: str) -> Optional[tuple[str, List[Dict]]]:
        """
        This method parses a single FSS NetCDF file to extract the experiment name, init_time, lead_time, and metric values for each valid time index. It uses helper functions to extract context from the filename and read metric values from the dataset. The result is a list of dictionaries, each representing a record with keys for initTime, leadTime, domain, thresh, window, and all metrics. If any step of the parsing fails (e.g. missing expected variables or filename patterns), the method returns None to indicate that the file could not be processed.

        Parameters:
            nc_file (str): Path to the FSS NetCDF file to parse.

        Returns:
            Optional[Tuple[str, List[Dict]]]: Tuple of ``(experiment, records)`` on
                success, or ``None`` when parsing fails.
        """
        from .utils import extract_lead_time_hours_from_path, parse_fss_filename_metadata

        context = self._extract_file_context(nc_file)
        if context is None:
            return None
        experiment, init_time = context

        lead_time = extract_lead_time_hours_from_path(nc_file)
        if lead_time is None:
            return None

        meta = parse_fss_filename_metadata(os.path.basename(nc_file))
        if meta is None:
            return None

        ds = xr.open_dataset(nc_file)
        fss_values, metric_values = self._parse_metric_values(ds)
        ds.close()

        records = [
            {
                "initTime": init_time,
                "leadTime": lead_time * (index + 1),
                "domain": meta["domain"],
                "thresh": meta["thresh"],
                "window": meta["window"],
                **{k: metric_values[k][index] for k in _METRIC_KEYS},
            }
            for index in range(len(fss_values))
        ]
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
        results_df = pd.DataFrame(records).sort_values(
            ["initTime", "leadTime", "domain", "thresh", "window"]
        )
        csv_path = os.path.join(csv_dir, f"{experiment}.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info("Wrote %s (%d records)", csv_path, len(results_df))

    def extract_fss_to_csv(
        self,
        output_dir: Optional[str] = None,
        csv_dir: Optional[str] = None,
    ) -> None:
        """
        Scan the output directory tree for FSS NetCDF files and write one CSV per experiment. All NetCDF files matching the ``**/ExtendedFC/**/*.nc`` glob pattern are discovered, and their filenames are parsed by parse_filename_metadata to extract domain, threshold, and window metadata. Lead times are extracted from the directory path via extract_lead_time_hours. Results are aggregated into a pandas DataFrame per experiment and written to ``<csv_dir>/<experiment>.csv``.

        Parameters:
            output_dir (str, optional): Root output directory to scan; defaults to the
                configured output_dir.
            csv_dir (str, optional): Directory where CSV files will be written; defaults
                to the configured csv_dir.

        Returns:
            None
        """
        import glob

        config = self.config
        output_dir = output_dir or config.resolve_path(config.output_dir)
        csv_dir = csv_dir or config.resolve_path(config.csv_dir)
        os.makedirs(csv_dir, exist_ok=True)

        nc_files = glob.glob(
            os.path.join(output_dir, "**/ExtendedFC/**/*.nc"), recursive=True
        )
        logger.info("Found %d FSS NetCDF files", len(nc_files))

        data: Dict[str, list] = defaultdict(list)
        for nc_file in nc_files:
            try:
                result = self._parse_records_from_nc_file(nc_file)
                if result is not None:
                    experiment, records = result
                    data[experiment].extend(records)
            except Exception:
                logger.warning("Error processing %s", nc_file, exc_info=True)

        for experiment, records in data.items():
            self._write_experiment_csv(experiment, records, csv_dir)
