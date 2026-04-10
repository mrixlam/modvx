#!/usr/bin/env python3

"""
Shared utility functions for MODvx.

This module contains helper functions for parsing datetime strings, formatting threshold values for filenames, iterating over valid times and forecast cycle starts, normalizing longitude coordinates, standardizing latitude/longitude dimension names, and extracting metadata from standardized FSS and contingency NetCDF filenames. These utilities are used across multiple components of the MODvx pipeline to ensure consistent handling of common tasks such as datetime parsing, filename conventions, and coordinate normalization. The functions are designed to be robust and flexible, supporting multiple input formats and providing clear error messages when inputs do not conform to expected patterns. By centralizing these common operations, this module promotes code reuse and maintainability across the MODvx codebase.

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
    This function attempts to parse a human-readable datetime string into a datetime object. It supports multiple common formats, including ``yyyymmddThh`` (e.g. ``20250613T00``) and ISO-8601 format (e.g. ``2025-06-13T00:00:00``). The function iterates through a predefined list of datetime formats and tries to parse the input string with each format until a successful parse is achieved. If none of the formats match the input string, a ValueError is raised with a descriptive message indicating the expected formats. This flexible parsing approach allows the function to handle various datetime string formats commonly used in meteorological data processing while providing clear feedback when inputs do not conform to expected patterns. 

    Parameters:
        datetime_str (str): Human-readable datetime string in ``yyyymmddThh`` or ISO-8601 format (e.g. ``"20250613T00"`` or ``"2025-06-13T00:00:00"``).

    Returns:
        datetime.datetime: Parsed datetime object corresponding to the input string.
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
        f"Invalid datetime format: {datetime_str!r}. Expected yyyymmddThh (e.g. 20250613T00) or ISO-8601 format (e.g. 2025-06-13T00:00:00)."
    )


def format_threshold_for_filename(threshold: float) -> str:
    """
    This function formats a threshold value for use in filenames by converting it to a string and replacing the decimal point with 'p'. This is necessary because decimal points can cause issues in filenames on some operating systems. For example, a threshold of 97.5 would be converted to "97p5". This function ensures that threshold values can be safely included in filenames while still being human-readable. 

    Parameters:
        threshold (float): Percentile threshold value (e.g., 97.5, 90.0).

    Returns:
        str: Filename-safe representation with the decimal point replaced by ``p`` (e.g., ``"97p5"`` for 97.5).
    """
    # Convert the threshold to a string and replace the decimal point with 'p' to create a filename-safe representation.
    return str(threshold).replace(".", "p")


def iterate_valid_times(start: datetime.datetime,
                        end: datetime.datetime,
                        step: datetime.timedelta,) -> Generator[datetime.datetime, None, None]:
    """
    This generator function yields valid time datetimes starting from the given start datetime up to but not including the end datetime, with a specified step interval. The function initializes the current time to the start datetime and enters a loop that continues until the current time reaches the end bound. Within the loop, it yields the current time and then increments it by the step interval. This allows callers to iterate over all valid times in a specified range with a consistent interval, which is useful for processing forecast data at regular time steps. The end datetime is exclusive, meaning that if the last valid time falls exactly on the end datetime, it will not be yielded. 

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


def iterate_forecast_cycle_starts(start: datetime.datetime,
                                  end: datetime.datetime,
                                  step: datetime.timedelta,) -> Generator[datetime.datetime, None, None]:
    """
    This generator function yields forecast cycle start datetimes from the specified start to end bounds, inclusive, with a given step interval. The function initializes the current cycle to the start datetime and enters a loop that continues until the current cycle exceeds the end datetime. Within the loop, it yields the current cycle start time and then increments it by the step interval. This allows callers to iterate over all forecast cycle start times within a specified range, which is essential for processing forecast data that is organized by initialization cycles. The end datetime is inclusive, meaning that if a cycle start falls exactly on the end datetime, it will be yielded as well. 

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


def normalize_longitude(data_array: xr.DataArray, 
                        target: str = "0_360") -> xr.DataArray:
    """
    This function normalizes the longitude coordinate of an xarray DataArray to a specified convention, either "0_360" or "-180_180". It first creates a copy of the longitude coordinate values to avoid modifying the original DataArray. Then, it uses vectorized operations to convert the longitude values to the target convention: for "0_360", it adds 360 to any negative values; for "-180_180", it subtracts 360 from any values greater than 180. After conversion, it assigns the modified longitude values back to the DataArray and sorts the data by longitude in ascending order. This ensures that the longitude coordinates are consistently represented according to the specified convention, which is important for accurate spatial analysis and visualization in meteorological data processing. 

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
    This function standardizes the latitude and longitude dimension names of an xarray DataArray to "latitude" and "longitude", respectively. It checks if the DataArray has dimensions named "lat" or "lon" and creates a renaming mapping accordingly. If any renaming is needed, it applies the renaming to the DataArray. This ensures that all DataArrays have consistent dimension names for latitude and longitude, which is important for interoperability across different datasets and components of the MODvx pipeline. By standardizing coordinate names, this function promotes code reuse and reduces the likelihood of errors due to inconsistent dimension naming conventions in different datasets. 

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
    This function extracts domain, threshold, and window metadata from a standardized FSS NetCDF filename. The expected filename pattern is ``modvx_metrics_type_neighborhood_<domain>_<Nh>h_indep_thresh<T>percent_window<W>.nc``, where threshold values may use ``p`` in place of ``.`` (e.g. ``97p5`` for 97.5). The function uses regular expressions to parse the filename and extract the relevant metadata components. If the filename does not conform to the expected pattern or if any of the required components cannot be extracted, the function returns ``None`` to indicate that parsing was unsuccessful. This allows calling code to handle unparseable filenames gracefully, such as by skipping them or logging a warning. When parsing is successful, the function returns a dictionary containing the extracted metadata values for domain, threshold, and window size, which can be used for further processing or analysis in the MODvx pipeline. 

    Parameters:
        filename (str): NetCDF filename to parse, with or without the ``.nc`` extension.

    Returns:
        dict or None: Dictionary with keys ``"domain"``, ``"thresh"``, and ``"window"`` if the pattern matches, or ``None`` if the filename cannot be parsed.
    """
    # Remove the .nc extension if present to simplify regex matching
    name = filename.replace(".nc", "")

    # Extract verification domain name using a regular expression
    domain_match = re.match(r"^modvx_metrics_type_neighborhood_(.+?)_\d+h_indep_", name)

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


def parse_contingency_filename_metadata(filename: str) -> dict | None:
    """
    This function extracts domain and threshold metadata from a standardized contingency NetCDF filename. The expected filename pattern is ``modvx_metrics_type_contingency_<domain>_<Nh>h_indep_thresh<T>percent.nc``, where threshold values may use ``p`` in place of ``.`` (e.g. ``97p5`` for 97.5). The function uses regular expressions to parse the filename and extract the relevant metadata components. If the filename does not conform to the expected pattern or if any of the required components cannot be extracted, the function returns ``None`` to indicate that parsing was unsuccessful. This allows calling code to handle unparseable filenames gracefully, such as by skipping them or logging a warning. When parsing is successful, the function returns a dictionary containing the extracted metadata values for domain and threshold, which can be used for further processing or analysis in the MODvx pipeline. 

    Parameters:
        filename (str): NetCDF filename to parse, with or without the ``.nc`` extension.

    Returns:
        dict or None: Dictionary with keys ``"domain"`` and ``"thresh"`` if the pattern matches, or ``None`` if the filename cannot be parsed.
    """
    # Remove the .nc extension if present to simplify regex matching
    name = filename.replace(".nc", "")

    # Extract verification domain name using a regular expression
    domain_match = re.match(r"^modvx_metrics_type_contingency_(.+?)_\d+h_indep_", name)

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

    # Return the extracted metadata as a dictionary
    return {"domain": domain, "thresh": thresh}


def extract_lead_time_hours_from_path(path_str: str) -> int | None:
    """
    This function extracts the lead-time in hours from a file path string that contains a token in the format ``pp##h``, where ``##`` represents the lead-time hour value. The function uses a regular expression to search for this pattern within the input string. If a match is found, it extracts the numeric portion representing the lead-time hours, converts it to an integer, and returns it. If no such pattern is found in the input string, the function returns ``None`` to indicate that the lead-time could not be extracted. This utility is useful for parsing file paths that encode lead-time information in their naming convention, allowing downstream code to programmatically determine the forecast lead-time associated with a given file. 

    Parameters:
        path_str (str): File path string potentially containing a ``pp##h`` lead-time token.

    Returns:
        int or None: Integer lead-time in hours extracted from the path, or ``None`` if not found.
    """
    # Extract the lead-time hour value using a regular expression that looks for the pattern "pp" followed by digits and "h"
    match = re.search(r"pp(\d+)h", path_str)

    # Return the extracted hour value as an integer
    return int(match.group(1)) if match else None
