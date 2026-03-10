"""
Shared utility functions for modvx.

Small, stateless helpers that are used across multiple modules and do not
belong to any single class.
"""

from __future__ import annotations

import datetime
import re
from typing import Generator

import numpy as np
import xarray as xr


def parse_datetime(s: str) -> datetime.datetime:
    """
    Parse a datetime string in one of several recognised formats into a Python datetime object.
    Accepted formats include the compact ``yyyymmddThh`` notation and ISO-8601 variants with
    full time components. The function tries each format sequentially and returns on the first
    successful parse. A descriptive ValueError is raised with the offending string when none
    of the formats match, guiding the user toward the expected input.

    Parameters:
        s (str): Human-readable datetime string in ``yyyymmddThh`` or ISO-8601 format
            (e.g. ``"20250613T00"`` or ``"2025-06-13T00:00:00"``).

    Returns:
        datetime.datetime: Parsed datetime object corresponding to the input string.

    Raises:
        ValueError: When the input string does not match any recognised datetime format.
    """
    for fmt in ("%Y%m%dT%H", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Invalid datetime format: {s!r}. Expected yyyymmddThh (e.g. 20250613T00)"
    )


def format_threshold_for_filename(threshold: float) -> str:
    """
    Convert a floating-point percentile threshold value to a filename-safe string.
    The decimal point is replaced by the letter ``p`` so that the value can be safely
    embedded in file and directory names without triggering path separator issues.
    For example, ``97.5`` becomes ``"97p5"`` and ``90.0`` becomes ``"90p0"``.
    The reverse transformation is applied when reading filenames in parse_filename_metadata.

    Parameters:
        threshold (float): Percentile threshold value (e.g., 97.5, 90.0).

    Returns:
        str: Filename-safe representation with the decimal point replaced by ``p``
            (e.g., ``"97p5"`` for 97.5).
    """
    return str(threshold).replace(".", "p")


def generate_valid_times(
    start: datetime.datetime,
    end: datetime.datetime,
    step: datetime.timedelta,
) -> Generator[datetime.datetime, None, None]:
    """
    Yield forecast valid-time datetimes over a specified range using a fixed step interval.
    Generation starts at *start* (inclusive) and stops before *end* (exclusive), advancing
    by *step* on each iteration. This generator is used to enumerate every valid time within
    a single forecast cycle. The calling code controls the range bounds to align with the
    configured forecast length and step size.

    Parameters:
        start (datetime.datetime): First valid time to yield (inclusive).
        end (datetime.datetime): Upper bound; the last yielded time is strictly less than this.
        step (datetime.timedelta): Time interval between consecutive valid times.

    Returns:
        Generator[datetime.datetime, None, None]: Generator yielding datetime objects for
            each valid time in the specified range.
    """
    t = start
    while t < end:
        yield t
        t += step


def generate_forecast_cycles(
    start: datetime.datetime,
    end: datetime.datetime,
    step: datetime.timedelta,
) -> Generator[datetime.datetime, None, None]:
    """
    Yield forecast cycle-start datetimes over a date range at a fixed interval.
    Both *start* and *end* are inclusive, so the final cycle is yielded if it falls
    exactly on *end*. This differs from generate_valid_times, which excludes the end bound.
    The function is used to enumerate initialisation times for multi-cycle verification
    experiments where the last cycle must be processed.

    Parameters:
        start (datetime.datetime): First cycle-start datetime to yield (inclusive).
        end (datetime.datetime): Last cycle-start datetime to yield (inclusive).
        step (datetime.timedelta): Interval between consecutive cycle-start datetimes.

    Returns:
        Generator[datetime.datetime, None, None]: Generator yielding each cycle-start datetime.
    """
    cycle = start
    while cycle <= end:
        yield cycle
        cycle += step


def normalize_longitude(da: xr.DataArray, target: str = "0_360") -> xr.DataArray:
    """
    Convert the longitude coordinate of a DataArray to either the [0, 360] or [-180, 180] convention.
    The function operates on the existing ``longitude`` coordinate values before reassigning them
    to the array. After conversion the DataArray is sorted along the longitude axis to maintain
    a monotonically increasing coordinate, which is required by interpolation and selection
    operations downstream. A ValueError is raised for unrecognised target conventions.

    Parameters:
        da (xr.DataArray): Input array with a coordinate named ``longitude``.
        target (str): Target longitude convention; either ``"0_360"`` (default) or ``"-180_180"``.

    Returns:
        xr.DataArray: DataArray with longitude values converted and sorted in ascending order.
    """
    lon = da.longitude.values.copy()
    if target == "0_360":
        lon = np.where(lon < 0, lon + 360, lon)
    elif target == "-180_180":
        lon = np.where(lon > 180, lon - 360, lon)
    else:
        raise ValueError(f"Unknown longitude target: {target!r}")
    da = da.assign_coords(longitude=lon).sortby("longitude")
    return da


def standardize_coords(da: xr.DataArray) -> xr.DataArray:
    """
    Rename abbreviated latitude/longitude dimension names to their full standard forms.
    Input arrays may use ``lat``/``lon`` (common in MPAS and many observational datasets)
    while the rest of the pipeline expects ``latitude``/``longitude``. This function
    performs that renaming transparently and is a no-op when the standard names are already
    present, making it safe to apply unconditionally to any incoming DataArray.

    Parameters:
        da (xr.DataArray): Input array potentially using abbreviated ``lat``/``lon`` dimension names.

    Returns:
        xr.DataArray: Array with dimensions renamed to ``latitude``/``longitude`` as needed.
    """
    renames = {}
    if "lat" in da.dims:
        renames["lat"] = "latitude"
    if "lon" in da.dims:
        renames["lon"] = "longitude"
    if renames:
        da = da.rename(renames)
    return da


def parse_filename_metadata(filename: str) -> dict | None:
    """
    Extract domain, threshold, and window metadata from a standardised FSS NetCDF filename.
    The expected filename pattern is ``<domain>_FSS_<Nh>_indep_thresh<T>percent_window<W>.nc``,
    where threshold values may use ``p`` in place of ``.`` (e.g. ``97p5`` for 97.5). This
    function is used during CSV extraction to parse result files without opening them. Returns
    ``None`` when the filename does not conform to the expected pattern.

    Parameters:
        filename (str): NetCDF filename to parse, with or without the ``.nc`` extension.

    Returns:
        dict or None: Dictionary with keys ``"domain"``, ``"thresh"``, and ``"window"`` if the
            pattern matches, or ``None`` if the filename cannot be parsed.
    """
    name = filename.replace(".nc", "")
    domain_m = re.match(r"^([^_]+(?:_[^_]+)*?)_FSS_", name)
    if not domain_m:
        return None
    domain = domain_m.group(1)

    thresh_m = re.search(r"thresh([\d.p]+)percent", name)
    if not thresh_m:
        return None
    thresh = thresh_m.group(1).replace("p", ".")

    window_m = re.search(r"window(\d+)", name)
    if not window_m:
        return None
    window = window_m.group(1)

    return {"domain": domain, "thresh": thresh, "window": window}


def extract_lead_time_hours(path_str: str) -> int | None:
    """
    Extract the lead-time hour value from a ``pp##h`` token embedded in a file path string.
    Lead times are encoded as directory components like ``pp12h`` or ``pp24h`` within the output
    directory hierarchy. This function locates the first such token and returns the integer hour
    value. Returns ``None`` when no matching token is found, allowing callers to skip files that
    do not follow the expected naming scheme.

    Parameters:
        path_str (str): File path string potentially containing a ``pp##h`` lead-time token.

    Returns:
        int or None: Integer lead-time in hours extracted from the path, or ``None`` if not found.
    """
    m = re.search(r"pp(\d+)h", path_str)
    return int(m.group(1)) if m else None
