"""
File I/O for modvx.

``FileManager`` centralises every filesystem operation — locating forecast
and observation files, loading/caching data, writing intermediate fields,
and saving FSS results as NetCDF.
"""

from __future__ import annotations

import datetime
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import xarray as xr

from .config import ModvxConfig
from .utils import format_threshold_for_filename, normalize_longitude, standardize_coords

logger = logging.getLogger(__name__)


class FileManager:
    """
    Centralise all filesystem operations for a modvx verification run.
    Responsibilities include locating forecast and observation files, loading and caching
    data with both in-memory and shared disk layers, writing intermediate debug fields
    as NetCDF, and persisting FSS results. The class is designed to be instantiated once
    per process and reused across all valid times in a cycle, allowing the in-memory
    observation cache to reduce redundant file reads.

    Parameters:
        config (ModvxConfig): Run configuration with directory paths, filename templates,
            and variable names.
    """

    def __init__(self, config: ModvxConfig) -> None:
        self.config = config
        self._obs_mem_cache: Dict[str, xr.DataArray] = {}

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    def get_forecast_filepath(
        self,
        valid_time: datetime.datetime,
        init_string: str,
    ) -> str:
        """
        Construct the full filesystem path to a native MPAS diagnostic file.
        MPAS diag files are organised under the experiment directory in an
        ``ExtendedFC/<init_string>/`` subdirectory, named using the valid-time timestamp
        with second precision (e.g. ``diag.2024-09-17_01.00.00.nc``). The path is built
        from the configured forecast directory, experiment name, and the provided temporal
        identifiers without any filesystem access.

        Parameters:
            valid_time (datetime.datetime): Forecast valid time, used to format the filename.
            init_string (str): Cycle initialisation string in ``YYYYmmddHH`` format
                (e.g. ``"2024091700"``).

        Returns:
            str: Full path to the MPAS diag NetCDF file for the specified valid time and cycle.
        """
        cfg = self.config
        ts = valid_time.strftime("%Y-%m-%d_%H.%M.%S")
        return str(
            Path(cfg.resolve_path(cfg.fcst_dir))
            / cfg.experiment_name
            / "ExtendedFC"
            / init_string
            / f"diag.{ts}.nc"
        )

    def get_observation_filepath(self, date_key: str) -> str:
        """
        Resolve the path to a FIMERG daily observation NetCDF file for a given date.
        The function iterates through the configured vintage preference list (e.g. FNL, LTE)
        and returns the path to the first file that actually exists on disk. This allows
        graceful fallback when the preferred final-run data is not yet available. If no
        qualifying vintage is found, the first-preference path is returned and a
        FileNotFoundError will be raised at load time.

        Parameters:
            date_key (str): Date string in ``YYYYmmdd`` format (e.g. ``"20240917"``).

        Returns:
            str: Full path to the FIMERG daily NetCDF file, using the first available vintage.
        """
        obs_dir = self.config.resolve_path(self.config.obs_dir)
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
        Map a FIMERG observation end-time to its zero-based hour index within the daily file.
        FIMERG daily files contain 24 hourly accumulation slices indexed 0–23, where index 0
        corresponds to the hour ending at 01:00 UTC (covering 00:00–01:00) and index 23
        corresponds to the hour ending at 00:00 UTC the following day (covering 23:00–00:00).
        This offset convention means that midnight (00:00) maps to index 23 rather than 0.

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
        Group a sequence of hourly observation end-times by their corresponding daily FIMERG file.
        The function iterates from *start_time* to *end_time* inclusive at *obs_interval* steps,
        assigning each time to the appropriate date key. Times at midnight (00:00 UTC) are
        assigned to the previous calendar day's file, consistent with FIMERG's end-of-hour
        timestamp convention. The result allows loading each daily file exactly once when
        accumulating observations over a multi-hour window.

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
        Load a binary verification-domain mask from a NetCDF file.
        After opening, dimension names are standardised to ``latitude``/``longitude``
        and longitudes are converted to the [0, 360] convention for consistency with
        observation data. The first non-coordinate variable in the file is treated as
        the mask variable. Raises descriptive errors when the file is missing or does
        not contain a suitable mask variable.

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
    # Forecast accumulation
    # ------------------------------------------------------------------

    def accumulate_forecasts(
        self,
        valid_time: datetime.datetime,
        init_string: str,
    ) -> xr.DataArray:
        """
        Compute accumulated precipitation for one forecast step from native MPAS diag files.
        MPAS diag files store cumulative ``rainc`` and ``rainnc`` from model initialisation,
        so the accumulation for a window is computed by subtracting the value at the window
        start from the value at the window end. The result is then remapped from the native
        unstructured MPAS mesh to a regular lat-lon grid at the configured resolution using
        the ``mpasdiag`` library.

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window
                (window end = valid_time + forecast_step).
            init_string (str): Cycle initialisation string in ``YYYYmmddHH`` format
                (e.g. ``"2024091700"``).

        Returns:
            xr.DataArray: Accumulated precipitation on a regular lat-lon grid in millimetres.
        """
        from .mpas_reader import load_mpas_precip, remap_to_latlon

        cfg = self.config
        grid_file = cfg.resolve_path(cfg.mpas_grid_file)
        resolution = cfg.mpas_remap_resolution

        window_end = valid_time + cfg.forecast_step

        end_file = self.get_forecast_filepath(window_end, init_string)
        start_file = self.get_forecast_filepath(valid_time, init_string)

        logger.debug("MPAS accum: %s → %s", start_file, end_file)

        end_precip = load_mpas_precip(end_file, grid_file)
        start_precip = load_mpas_precip(start_file, grid_file)

        accum_mesh = end_precip - start_precip
        accum_mesh.attrs["units"] = "mm"
        accum_mesh.attrs["long_name"] = (
            f"{int(cfg.forecast_step_hours)}h accumulated precipitation"
        )

        # Remap unstructured mesh → regular lat-lon
        remapped = remap_to_latlon(
            accum_mesh, end_file, grid_file, resolution,
        )

        logger.debug(
            "MPAS accumulated %dh precip from %s, remapped to %.3f° grid",
            cfg.forecast_step_hours, valid_time, resolution,
        )
        return remapped

    # ------------------------------------------------------------------
    # Observation accumulation (with shared cache)
    # ------------------------------------------------------------------

    def _obs_cache_key(self, valid_time: datetime.datetime) -> str:
        """
        Generate a deterministic string cache key for an accumulated observation field.
        The key encodes the valid time and forecast step duration, ensuring that cache
        entries are unique per (valid_time, accumulation_length) combination. The same
        key is used to check both the in-memory dict cache and the shared disk cache
        directory, making it straightforward to coordinate cache reads across workers.

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
        Load and accumulate FIMERG observation data with multi-level caching.
        Three cache layers are checked in order: (1) an in-memory dict per process, which
        avoids re-loading data when the same valid time appears in multiple cycles; (2) a
        shared disk cache directory containing pre-computed NetCDF files, which allows
        multiprocessing workers to share work; and (3) the original FIMERG daily source
        files as the final fallback. Successfully loaded data is stored in both cache
        layers for future use.

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window, aligned with
                the forecast valid time.

        Returns:
            xr.DataArray: Accumulated hourly precipitation sum over the forecast step period.
        """
        key = self._obs_cache_key(valid_time)

        # --- 1. In-memory cache ---
        if key in self._obs_mem_cache:
            logger.debug("Obs cache hit (memory): %s", key)
            return self._obs_mem_cache[key]

        # --- 2. Shared disk cache ---
        cache_dir = self.config.cache_dir
        if cache_dir:
            disk_path = os.path.join(cache_dir, f"{key}.nc")
            if os.path.exists(disk_path):
                logger.debug("Obs cache hit (disk): %s", disk_path)
                ds = xr.load_dataset(disk_path)
                da = ds[list(ds.data_vars)[0]]
                self._obs_mem_cache[key] = da
                return da

        # --- 3. Load from source FIMERG files ---
        da = self._accumulate_observations_raw(valid_time)

        # Store in caches
        self._obs_mem_cache[key] = da
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            disk_path = os.path.join(cache_dir, f"{key}.nc")
            tmp_path = f"{disk_path}.tmp.{os.getpid()}"
            da.to_netcdf(tmp_path)
            os.rename(tmp_path, disk_path)  # atomic on POSIX
            logger.debug("Obs cached to disk: %s", disk_path)

        return da

    def _accumulate_observations_raw(
        self,
        valid_time: datetime.datetime,
    ) -> xr.DataArray:
        """
        Load and sum hourly FIMERG observation slices over one forecast step from source files.
        This internal method performs the actual file I/O without any caching, making it the
        fallback called by accumulate_observations when no cached result is available. Hourly
        slices are grouped by daily file to minimise the number of NetCDF files opened. The
        method asserts that at least one observation slice was accumulated to catch silent
        data gaps early.

        Parameters:
            valid_time (datetime.datetime): Start of the accumulation window
                (window end = valid_time + forecast_step).

        Returns:
            xr.DataArray: Accumulated hourly precipitation sum over the forecast step period.
        """
        cfg = self.config
        start_time = valid_time + cfg.observation_interval
        end_time = valid_time + cfg.forecast_step

        times_by_date = self.group_observation_times_by_date(
            start_time, end_time, cfg.observation_interval
        )

        accumulated: Optional[xr.DataArray] = None
        cnt = 0

        for date_key, times in sorted(times_by_date.items()):
            daily_file = self.get_observation_filepath(date_key)
            logger.debug("Loading obs file: %s", daily_file)
            ds = xr.load_dataset(daily_file)

            for t in times:
                idx = self.get_observation_hour_index(t)
                arr = ds[cfg.obs_var_name].isel(time=idx)
                accumulated = arr if accumulated is None else accumulated + arr
                cnt += 1

        assert accumulated is not None, "No observation data for accumulation"
        logger.debug("Accumulated %d obs hours from %s", cnt, valid_time)
        return accumulated

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
        Save regridded and masked forecast and observation precipitation fields to a debug NetCDF.
        The output file contains three variables: ``forecast``, ``observation``, and
        ``difference``, each with a time dimension encoded as seconds since the Unix epoch.
        Files are written to a per-cycle subdirectory under the configured debug directory
        and are only produced when the ``save_intermediate`` flag is enabled. zlib
        compression is applied at the configured level to minimise disk usage.

        Parameters:
            forecast_da (xr.DataArray): Regridded and masked forecast precipitation field.
            observation_da (xr.DataArray): Regridded and masked observation precipitation field.
            cycle_start (datetime.datetime): Initialisation time of the current forecast cycle.
            valid_time (datetime.datetime): Start of the current accumulation window.

        Returns:
            None
        """
        cfg = self.config
        init_str = cycle_start.strftime("%Y%m%d%H")
        end_time = valid_time + cfg.forecast_step
        valid_str = end_time.strftime("%Y%m%d%H%M")

        odir = os.path.join(
            cfg.resolve_path(cfg.debug_dir), cfg.experiment_name, init_str, "precip"
        )
        os.makedirs(odir, exist_ok=True)

        time_val = (end_time - datetime.datetime(1970, 1, 1)).total_seconds()
        fcst_t = forecast_da.expand_dims({"time": [time_val]})
        obs_t = observation_da.expand_dims({"time": [time_val]})

        ds = xr.Dataset(
            {"forecast": fcst_t, "observation": obs_t, "difference": fcst_t - obs_t}
        )
        ds["time"].attrs.update(
            standard_name="time",
            long_name="Time",
            units="seconds since 1970-01-01",
            calendar="standard",
        )
        ds.attrs.update(
            experiment=cfg.experiment_name,
            cycle_start=cycle_start.strftime("%Y-%m-%d %H:%M:%S"),
            valid_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        enc = {v: {"zlib": True, "complevel": cfg.compression_level} for v in ds.data_vars}
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
        Save binary exceedance mask fields to a debug NetCDF file for a specific threshold.
        The output file contains two variables, ``forecast_binary`` and ``observation_binary``,
        representing the 0/1/NaN exceedance masks produced by PerfMetrics.generate_binary_mask.
        Files are named using the threshold value and valid-time string and written to a
        per-cycle binary subdirectory under the debug directory. This output is only produced
        when the ``save_intermediate`` flag is enabled in the configuration.

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
        cfg = self.config
        init_str = cycle_start.strftime("%Y%m%d%H")
        end_time = valid_time + cfg.forecast_step
        valid_str = end_time.strftime("%Y%m%d%H%M")

        odir = os.path.join(
            cfg.resolve_path(cfg.debug_dir), cfg.experiment_name, init_str, "binary"
        )
        os.makedirs(odir, exist_ok=True)

        time_val = (end_time - datetime.datetime(1970, 1, 1)).total_seconds()
        fb = forecast_binary.expand_dims({"time": [time_val]})
        ob = observation_binary.expand_dims({"time": [time_val]})

        ds = xr.Dataset({"forecast_binary": fb, "observation_binary": ob})
        ds["time"].attrs.update(
            standard_name="time",
            long_name="Time",
            units="seconds since 1970-01-01",
            calendar="standard",
        )
        ds.attrs.update(
            experiment=cfg.experiment_name,
            cycle_start=cycle_start.strftime("%Y-%m-%d %H:%M:%S"),
            valid_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            threshold_percentile=threshold,
        )

        enc = {v: {"zlib": True, "complevel": cfg.compression_level} for v in ds.data_vars}
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
        Persist verification metrics for a single (cycle, region, threshold, window) combination.
        Each element of metrics_list is a dictionary of metric values (fss, pod, far, csi, fbias,
        ets) for one valid time. Files are written to the output directory hierarchy under
        ``<output_dir>/<experiment>/ExtendedFC/<init_string>/pp<step>h/`` with a standardised
        filename encoding the region name, threshold percentile, and window size. All metric
        arrays are stored as variables inside a single NetCDF dataset so that extract_fss_to_csv
        can reconstruct the full record.

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
        cfg = self.config
        istr = cycle_start.strftime("%Y%m%d%H")
        step_h = int(cfg.forecast_step.total_seconds() / 3600)
        pp_dir = f"pp{step_h}h"

        odir = os.path.join(
            cfg.resolve_path(cfg.output_dir),
            cfg.experiment_name,
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

    def extract_fss_to_csv(
        self,
        output_dir: Optional[str] = None,
        csv_dir: Optional[str] = None,
    ) -> None:
        """
        Scan the output directory tree for FSS NetCDF files and write one CSV per experiment.
        All NetCDF files matching the ``**/ExtendedFC/**/*.nc`` glob pattern are discovered,
        and their filenames are parsed by parse_filename_metadata to extract domain, threshold,
        and window metadata. Lead times are extracted from the directory path via
        extract_lead_time_hours. Results are aggregated into a pandas DataFrame per experiment
        and written to ``<csv_dir>/<experiment>.csv``.

        Parameters:
            output_dir (str, optional): Root output directory to scan; defaults to the
                configured output_dir.
            csv_dir (str, optional): Directory where CSV files will be written; defaults
                to the configured csv_dir.

        Returns:
            None
        """
        import glob
        import re

        import pandas as pd

        from .utils import extract_lead_time_hours, parse_filename_metadata

        cfg = self.config
        output_dir = output_dir or cfg.resolve_path(cfg.output_dir)
        csv_dir = csv_dir or cfg.resolve_path(cfg.csv_dir)
        os.makedirs(csv_dir, exist_ok=True)

        pattern = os.path.join(output_dir, "**/ExtendedFC/**/*.nc")
        nc_files = glob.glob(pattern, recursive=True)
        logger.info("Found %d FSS NetCDF files", len(nc_files))

        data: Dict[str, list] = defaultdict(list)

        for nc_file in nc_files:
            try:
                parts = Path(nc_file).parts
                out_idx = parts.index("output")
                experiment = parts[out_idx + 1]

                init_time: Optional[str] = None
                for p in parts:
                    if re.match(r"^\d{10}$", p):
                        init_time = p
                        break
                if init_time is None:
                    continue

                lead_time = extract_lead_time_hours(nc_file)
                if lead_time is None:
                    continue

                meta = parse_filename_metadata(os.path.basename(nc_file))
                if meta is None:
                    continue

                ds = xr.open_dataset(nc_file)

                # Support both new multi-metric format and legacy single-array format
                metric_keys = ["fss", "pod", "far", "csi", "fbias", "ets"]
                if "fss" in ds:
                    fss_vals = ds["fss"].values
                    metric_vals = {
                        k: ds[k].values if k in ds else [float("nan")] * len(fss_vals)
                        for k in metric_keys
                    }
                else:
                    fss_vals = ds["__xarray_dataarray_variable__"].values
                    metric_vals = {
                        "fss": fss_vals,
                        **{k: [float("nan")] * len(fss_vals) for k in metric_keys if k != "fss"},
                    }
                ds.close()

                for idx in range(len(fss_vals)):
                    record = {
                        "initTime": init_time,
                        "leadTime": lead_time * (idx + 1),
                        "domain": meta["domain"],
                        "thresh": meta["thresh"],
                        "window": meta["window"],
                    }
                    for k in metric_keys:
                        record[k] = metric_vals[k][idx]
                    data[experiment].append(record)
            except Exception:
                logger.warning("Error processing %s", nc_file, exc_info=True)

        for experiment, records in data.items():
            df = pd.DataFrame(records).sort_values(
                ["initTime", "leadTime", "domain", "thresh", "window"]
            )
            csv_path = os.path.join(csv_dir, f"{experiment}.csv")
            df.to_csv(csv_path, index=False)
            logger.info("Wrote %s (%d records)", csv_path, len(df))
