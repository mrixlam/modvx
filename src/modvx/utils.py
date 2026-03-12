#!/usr/bin/env python3

"""
Shared utility functions for MODvx.

This module contains helper functions for parsing datetime strings, formatting threshold values for filenames, iterating over valid times and forecast cycle starts, normalizing longitude coordinates, standardizing coordinate names, and extracting metadata from filenames. These utilities are used across multiple components of the MODvx pipeline to ensure consistent handling of common tasks related to time management, file naming conventions, and data array manipulation. By centralizing these functions, we promote code reuse and maintainability throughout the verification workflow.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
from __future__ import annotations

import re
import datetime
import numpy as np
import xarray as xr
from typing import Generator


def parse_datetime_string(datetime_str: str) -> datetime.datetime:
    """
    Parse a datetime string in one of several recognised formats into a Python datetime object. Accepted formats include the compact ``yyyymmddThh`` notation and ISO-8601 variants with full time components. The function tries each format sequentially and returns on the first successful parse. A descriptive ValueError is raised with the offending string when none of the formats match, guiding the user toward the expected input.

    Parameters:
        datetime_str (str): Human-readable datetime string in ``yyyymmddThh`` or ISO-8601 format (e.g. ``"20250613T00"`` or ``"2025-06-13T00:00:00"``).

    Returns:
        datetime.datetime: Parsed datetime object corresponding to the input string.

    Raises:
        ValueError: When the input string does not match any recognised datetime format.
    """
    # Define a list of datetime formats 
    datetime_list = [
        "%Y%m%dT%H",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]

    # Parse the input string using each format in the list until a successful parse is achieved. 
    for fmt in datetime_list:
        try:
            return datetime.datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue

    # Return a ValueError with a descriptive message if none of the formats match the input string
    raise ValueError(
        f"Invalid datetime format: {datetime_str!r}. Expected yyyymmddThh (e.g. 20250613T00)"
    )


def format_threshold_for_filename(threshold: float) -> str:
    """
    Convert a floating-point percentile threshold value to a filename-safe string. The decimal point is replaced by the letter ``p`` so that the value can be safely embedded in file and directory names without triggering path separator issues. For example, ``97.5`` becomes ``"97p5"`` and ``90.0`` becomes ``"90p0"``. The reverse transformation is applied when reading filenames in parse_filename_metadata.

    Parameters:
        threshold (float): Percentile threshold value (e.g., 97.5, 90.0).

    Returns:
        str: Filename-safe representation with the decimal point replaced by ``p`` (e.g., ``"97p5"`` for 97.5).
    """
    # Convert the threshold to a string and replace the decimal point with 'p' to create a filename-safe representation.
    return str(threshold).replace(".", "p")


def iterate_valid_times(
    start: datetime.datetime,
    end: datetime.datetime,
    step: datetime.timedelta,
) -> Generator[datetime.datetime, None, None]:
    """
    Yield forecast valid-time datetimes over a specified range using a fixed step interval. Generation starts at *start* (inclusive) and stops before *end* (exclusive), advancing by *step* on each iteration. This generator is used to enumerate every valid time within a single forecast cycle. The calling code controls the range bounds to align with the configured forecast length and step size.

    Parameters:
        start (datetime.datetime): First valid time to yield (inclusive).
        end (datetime.datetime): Upper bound; the last yielded time is strictly less than this.
        step (datetime.timedelta): Time interval between consecutive valid times.

    Returns:
        Generator[datetime.datetime, None, None]: Generator yielding datetime objects for each valid time in the specified range.
    """
    # Initialize the current time to the start datetime
    current_time = start

    # Loop until the current time reaches the end bound, yielding each valid time along the way. 
    while current_time < end:
        yield current_time
        current_time += step


def iterate_forecast_cycle_starts(
    start: datetime.datetime,
    end: datetime.datetime,
    step: datetime.timedelta,
) -> Generator[datetime.datetime, None, None]:
    """
    Yield forecast cycle-start datetimes over a date range at a fixed interval. Both *start* and *end* are inclusive, so the final cycle is yielded if it falls exactly on *end*. This differs from generate_valid_times, which excludes the end bound. The function is used to enumerate initialisation times for multi-cycle verification experiments where the last cycle must be processed.

    Parameters:
        start (datetime.datetime): First cycle-start datetime to yield (inclusive).
        end (datetime.datetime): Last cycle-start datetime to yield (inclusive).
        step (datetime.timedelta): Interval between consecutive cycle-start datetimes.

    Returns:
        Generator[datetime.datetime, None, None]: Generator yielding each cycle-start datetime.
    """
    # Initialize the current cycle to the start datetime
    current_cycle = start

    # Loop until the current cycle exceeds the end datetime, yielding each cycle-start datetime along the way. 
    while current_cycle <= end:
        yield current_cycle
        current_cycle += step


def normalize_longitude(data_array: xr.DataArray, target: str = "0_360") -> xr.DataArray:
    """
    Convert the longitude coordinate of a DataArray to either the [0, 360] or [-180, 180] convention. The function operates on the existing ``longitude`` coordinate values before reassigning them to the array. After conversion the DataArray is sorted along the longitude axis to maintain a monotonically increasing coordinate, which is required by interpolation and selection operations downstream. A ValueError is raised for unrecognised target conventions.

    Parameters:
        data_array (xr.DataArray): Input array with a coordinate named ``longitude``.
        target (str): Target longitude convention; either ``"0_360"`` (default) or ``"-180_180"``.

    Returns:
        xr.DataArray: DataArray with longitude values converted and sorted in ascending order.
    """
    # Make a copy of the longitude coordinate values to avoid accidental modification of the original DataArray
    lon = data_array.longitude.values.copy()

    # Convert longitude values to the target convention using vectorized operations for efficiency. 
    if target == "0_360":
        lon = np.where(lon < 0, lon + 360, lon)
    elif target == "-180_180":
        lon = np.where(lon > 180, lon - 360, lon)
    else:
        raise ValueError(f"Unknown longitude target: {target!r}")

    # Assign the converted longitude values back to the DataArray 
    data_array = data_array.assign_coords(longitude=lon).sortby("longitude")

    # Return standardized data array
    return data_array


def standardize_coords(data_array: xr.DataArray) -> xr.DataArray:
    """
    Rename abbreviated latitude/longitude dimension names to their full standard forms. Input arrays may use ``lat``/``lon`` (common in MPAS and many observational datasets) while the rest of the pipeline expects ``latitude``/``longitude``. This function performs that renaming transparently and is a no-op when the standard names are already present, making it safe to apply unconditionally to any incoming DataArray.

    Parameters:
        data_array (xr.DataArray): Input array potentially using abbreviated ``lat``/``lon`` dimension names.

    Returns:
        xr.DataArray: Array with dimensions renamed to ``latitude``/``longitude`` as needed.
    """
    # Initialize an empty dictionary to hold any necessary renaming mappings
    rename_map: dict = {}

    # Standardize latitude dimension name if necessary
    if "lat" in data_array.dims:
        rename_map["lat"] = "latitude"

    # Standardize longitude dimension name if necessary
    if "lon" in data_array.dims:
        rename_map["lon"] = "longitude"

    # Perform the renaming as needed
    if rename_map:
        data_array = data_array.rename(rename_map)

    # Return standardized DataArray with latitude and longitude dimensions renamed as needed.
    return data_array


def parse_fss_filename_metadata(filename: str) -> dict | None:
    """
    Extract domain, threshold, and window metadata from a standardised FSS NetCDF filename. The expected filename pattern is ``modvx_metrics_<domain>_<Nh>h_indep_thresh<T>percent_window<W>.nc``, where threshold values may use ``p`` in place of ``.`` (e.g. ``97p5`` for 97.5). This function is used during CSV extraction to parse result files without opening them. Returns ``None`` when the filename does not conform to the expected pattern.

    Parameters:
        filename (str): NetCDF filename to parse, with or without the ``.nc`` extension.

    Returns:
        dict or None: Dictionary with keys ``"domain"``, ``"thresh"``, and ``"window"`` if the pattern matches, or ``None`` if the filename cannot be parsed.
    """
    # Remove the .nc extension if present to simplify regex matching
    name = filename.replace(".nc", "")

    # Extract verification domain name using a regular expression
    domain_match = re.match(r"^modvx_metrics_(.+?)_\d+h_indep_", name)

    # Return None if no matching domain name found
    if not domain_match:
        return None

    # Extract the domain name
    domain = domain_match.group(1)

    # Extract threshold value using a regular expression
    thresh_match = re.search(r"thresh([\d.p]+)percent", name)

    # Return None if no matching threshold value found
    if not thresh_match:
        return None

    # Extract the threshold information
    thresh = thresh_match.group(1).replace("p", ".")

    # Extract the window size value using a regular expression
    window_match = re.search(r"window(\d+)", name)

    # Return None if no matching window value found 
    if not window_match:
        return None

    # Extract the window size 
    window = window_match.group(1)

    # Return the extracted metadata as a dictionary
    return {"domain": domain, "thresh": thresh, "window": window}


def extract_lead_time_hours_from_path(path_str: str) -> int | None:
    """
    Extract the lead-time hour value from a ``pp##h`` token embedded in a file path string. Lead times are encoded as directory components like ``pp12h`` or ``pp24h`` within the output directory hierarchy. This function locates the first such token and returns the integer hour value. Returns ``None`` when no matching token is found, allowing callers to skip files that do not follow the expected naming scheme.

    Parameters:
        path_str (str): File path string potentially containing a ``pp##h`` lead-time token.

    Returns:
        int or None: Integer lead-time in hours extracted from the path, or ``None`` if not found.
    """
    # Extract the lead-time hour value using a regular expression that looks for the pattern "pp" followed by digits and "h"
    match = re.search(r"pp(\d+)h", path_str)

    # Return the extracted hour value as an integer
    return int(match.group(1)) if match else None
