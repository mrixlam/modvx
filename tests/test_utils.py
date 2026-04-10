#!/usr/bin/env python3

"""
Unit tests for MODvx utilities module.

This module verifies the small, stateless helper functions used throughout the modvx pipeline, including datetime parsing, filename metadata extraction, longitude normalization, and grid/time sequence generators. Tests exercise both normal and edge cases to ensure correct parsing, coordinate normalization, and iterator semantics. The suite is self-contained and uses lightweight fixtures so helpers can be validated in isolation without external data dependencies.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

import datetime

import pytest

from modvx.utils import (
    extract_lead_time_hours_from_path,
    format_threshold_for_filename,
    iterate_forecast_cycle_starts,
    iterate_valid_times,
    normalize_longitude,
    parse_datetime_string,
    parse_fss_filename_metadata,
    standardize_coords,
)

import numpy as np
import xarray as xr


class TestParseDatetime:
    """ Tests for parse_datetime_string verifying correct parsing of both compact canonical and ISO 8601 datetime formats, as well as error handling for invalid strings. """

    def test_canonical(self: "TestParseDatetime") -> None:
        """
        This test verifies that parse_datetime_string correctly parses a compact canonical datetime string in the format 'YYYYMMDDTHH' and returns a datetime object with the expected year, month, day, hour, and zeroed minute and second components. The canonical format is commonly used in NetCDF metadata and configuration files, so correct parsing is essential for constructing forecast cycle generators and valid time sequences. The test confirms that all components are correctly extracted and that the resulting datetime object matches the expected value. 

        Parameters:
            None

        Returns:
            None
        """
        assert parse_datetime_string("20250613T00") == datetime.datetime(2025, 6, 13, 0, 0)

    def test_iso(self: "TestParseDatetime") -> None:
        """
        This test confirms that parse_datetime_string can also parse an ISO 8601 datetime string in the format 'YYYY-MM-DDTHH:MM:SS' and returns a datetime object with the correct components. The ISO format is widely used in configuration files and metadata, so supporting it ensures flexibility in input formats. The test checks that the function correctly handles the additional delimiters and that the resulting datetime object matches the expected value with the correct year, month, day, hour, minute, and second. 

        Parameters:
            None

        Returns:
            None
        """
        assert parse_datetime_string("2024-09-23T06:00:00") == datetime.datetime(2024, 9, 23, 6, 0)

    def test_invalid_raises(self: "TestParseDatetime") -> None:
        """
        This test ensures that parse_datetime_string raises a ValueError when given an invalid datetime string that does not match either the canonical or ISO formats. Proper error handling is crucial to prevent silent failures and to allow callers to catch and handle parsing errors gracefully. The test uses a clearly invalid string and confirms that a ValueError is raised, indicating that the function correctly identifies unparseable input. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError):
            parse_datetime_string("not-a-date")


class TestFormatThreshold:
    """ Tests for format_threshold_for_filename verifying correct conversion of float threshold values to filename-safe strings. """

    def test_integer_like(self: "TestFormatThreshold") -> None:
        """
        This test verifies that format_threshold_for_filename correctly formats a float value that is mathematically an integer (e.g., 90.0) by replacing the decimal point with 'p' to produce a filename-safe string. This ensures that thresholds like 90.0, which are commonly used in percentile-based verification, are consistently encoded as '90p0' in output filenames. The test confirms that the function produces the expected string output for this case, which is critical for downstream code that relies on parsing these filename tokens to reconstruct the original float values. 

        Parameters:
            None

        Returns:
            None
        """
        assert format_threshold_for_filename(90.0) == "90p0"

    def test_decimal(self: "TestFormatThreshold") -> None:
        """
        This test confirms that format_threshold_for_filename correctly formats a float value with a non-zero decimal component (e.g., 97.5) by replacing the decimal point with 'p' to produce a filename-safe string. This ensures that thresholds like 97.5, which are commonly used in percentile-based verification, are consistently encoded as '97p5' in output filenames. The test checks that the function produces the expected string output for this case, which is important for downstream code that relies on parsing these filename tokens to reconstruct the original float values accurately. 

        Parameters:
            None

        Returns:
            None
        """
        assert format_threshold_for_filename(97.5) == "97p5"


class TestGenerators:
    """ Tests for generate_valid_times and generate_forecast_cycles verifying sequence length, endpoint inclusion, and correct step-based iteration."""

    def test_valid_times(self: "TestGenerators") -> None:
        """
        This test verifies that iterate_valid_times produces a sequence of datetimes starting from the specified start time, incrementing by the given step, and excluding the end time. The test confirms that the resulting list of valid times has the expected length based on the start, end, and step parameters, and that the first and last elements of the sequence are correct according to the exclusive end convention. This ensures that valid time generation operates as intended for constructing forecast verification runs with the correct set of valid times.

        Parameters:
            None

        Returns:
            None
        """
        start = datetime.datetime(2025, 1, 1, 0)
        end = datetime.datetime(2025, 1, 1, 6)
        step = datetime.timedelta(hours=2)
        result = list(iterate_valid_times(start, end, step))
        assert len(result) == 3
        assert result[0] == start
        assert result[-1] == datetime.datetime(2025, 1, 1, 4)

    def test_forecast_cycles_inclusive(self: "TestGenerators") -> None:
        """
        This test confirms that iterate_forecast_cycle_starts produces a sequence of datetimes starting from the specified start time, incrementing by the given step, and including the end time if it falls on a valid cycle. The test checks that the resulting list of forecast cycle starts has the expected length based on the start, end, and step parameters, and that the last element of the sequence is equal to the end time when it is included. This ensures that forecast cycle generation correctly handles inclusive endpoints when they align with the stepping interval, which is important for constructing complete sets of forecast cycles for verification runs.

        Parameters:
            None

        Returns:
            None
        """
        start = datetime.datetime(2025, 1, 1, 0)
        end = datetime.datetime(2025, 1, 3, 0)
        step = datetime.timedelta(hours=24)
        result = list(iterate_forecast_cycle_starts(start, end, step))
        assert len(result) == 3  # day 1, 2, 3


class TestNormalizeLongitude:
    """ Tests for normalize_longitude verifying correct remapping of longitude coordinates to the specified target convention. """

    def test_negative_to_0_360(self: "TestNormalizeLongitude") -> None:
        """
        This test verifies that normalize_longitude correctly remaps longitude coordinates from the [-180, 180) range to the [0, 360) range when the '0_360' convention is requested. This remapping is necessary for compatibility with certain datasets and plotting libraries that expect longitudes in the [0, 360) format. The test constructs a DataArray with longitude coordinates that include negative values and confirms that after normalization, all longitude values are within the expected non-negative range. This ensures that the function correctly handles the remapping logic for negative longitudes. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones((2, 4)),
            dims=["latitude", "longitude"],
            coords={"latitude": [0, 1], "longitude": [-180, -90, 0, 90]},
        )
        result = normalize_longitude(da, "0_360")
        assert float(result.longitude.min()) >= 0
        assert float(result.longitude.max()) <= 360

    def test_positive_to_neg180(self: "TestNormalizeLongitude") -> None:
        """
        This test confirms that normalize_longitude correctly remaps longitude coordinates from the [0, 360) range to the [-180, 180) range when the '-180_180' convention is requested. This remapping is necessary for compatibility with certain datasets and plotting libraries that expect longitudes in the [-180, 180) format. The test constructs a DataArray with longitude coordinates that include values greater than 180 and confirms that after normalization, all longitude values are within the expected range of -180 to 180. This ensures that the function correctly handles the remapping logic for positive longitudes. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones((2, 3)),
            dims=["latitude", "longitude"],
            coords={"latitude": [0, 1], "longitude": [0, 180, 270]},
        )
        result = normalize_longitude(da, "-180_180")
        assert float(result.longitude.min()) >= -180
        assert float(result.longitude.max()) <= 180


class TestStandardizeCoords:
    """ Tests for standardize_coords verifying dimension renaming from short-form to canonical long-form coordinate names and no-op behavior when names are already correct. """

    def test_rename_lat_lon(self: "TestStandardizeCoords") -> None:
        """
        This test verifies that standardize_coords correctly renames dimensions from 'lat' to 'latitude' and 'lon' to 'longitude' when those short-form names are present. This renaming is important for ensuring compatibility with libraries and functions that expect the canonical long-form coordinate names. The test constructs a DataArray with 'lat' and 'lon' dimensions and confirms that after calling standardize_coords, the resulting DataArray has 'latitude' and 'longitude' in its dimensions. This ensures that the function correctly identifies and renames the relevant dimensions. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.ones((3, 3)), dims=["lat", "lon"])
        result = standardize_coords(da)
        assert "latitude" in result.dims
        assert "longitude" in result.dims

    def test_noop_when_standard(self: "TestStandardizeCoords") -> None:
        """
        This test confirms that standardize_coords does not modify dimension names when they are already in the canonical long-form format. If the input DataArray already has 'latitude' and 'longitude' dimensions, the function should return it unchanged without renaming. The test constructs a DataArray with 'latitude' and 'longitude' dimensions and confirms that after calling standardize_coords, the resulting DataArray still has 'latitude' and 'longitude' in its dimensions, indicating that no unnecessary renaming occurred. This ensures that the function behaves as a no-op when the coordinate names are already correct. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.ones((3, 3)), dims=["latitude", "longitude"])
        result = standardize_coords(da)
        assert "latitude" in result.dims


class TestParseFilenameMetadata:
    """ Tests for parse_filename_metadata verifying regex-based extraction of domain, threshold, and window tokens from FSS output filenames. """

    def test_standard(self: "TestParseFilenameMetadata") -> None:
        """
        This test verifies that parse_fss_filename_metadata correctly extracts the domain, threshold, and window size components from a standard FSS output filename that follows the expected naming convention. The test uses a representative filename containing the 'global' domain, a threshold of '90.0' encoded as '90p0', and a window size of '3'. The function should return a dictionary with these values correctly parsed and formatted, confirming that the regex-based extraction logic is functioning as intended for typical FSS output filenames. 

        Parameters:
            None

        Returns:
            None
        """
        m = parse_fss_filename_metadata(
            "modvx_metrics_type_neighborhood_global_12h_indep_thresh90p0percent_window3.nc"
        )
        assert m == {"domain": "global", "thresh": "90.0", "window": "3"}

    def test_decimal_thresh(self: "TestParseFilenameMetadata") -> None:
        """
        This test confirms that parse_fss_filename_metadata correctly extracts and formats a decimal threshold value from an FSS output filename. The test uses a filename containing the 'tropics' domain, a threshold of '97.5' encoded as '97p5', and a window size of '11'. The function should return a dictionary with the threshold value correctly parsed and converted back to the standard decimal string format, confirming that the regex-based extraction logic can handle thresholds with non-zero decimal components as expected. 

        Parameters:
            None

        Returns:
            None
        """
        m = parse_fss_filename_metadata(
            "modvx_metrics_type_neighborhood_tropics_6h_indep_thresh97p5percent_window11.nc"
        )
        assert m is not None
        assert m["thresh"] == "97.5"

    def test_bad_filename(self: "TestParseFilenameMetadata") -> None:
        """
        This test ensures that parse_fss_filename_metadata returns None when given a filename that does not match the expected pattern for FSS output files. The function relies on specific tokens in the filename to extract metadata, so a filename that lacks these components should result in a None return value rather than raising an exception. This allows callers to gracefully skip files that do not conform to the expected naming convention without crashing. The test uses a simple filename with no recognizable patterns and confirms that the function returns None as expected. 

        Parameters:
            None

        Returns:
            None
        """
        assert parse_fss_filename_metadata("random_file.nc") is None


class TestExtractLeadTime:
    """ Tests for extract_lead_time_hours_from_path verifying correct regex extraction of lead time hours from output paths containing 'ppNh' directory components."""

    def test_pp12h(self: "TestExtractLeadTime") -> None:
        """
        This test verifies that extract_lead_time_hours_from_path correctly extracts the lead time in hours from a path that contains a 'ppNh' directory component, where N is the number of hours. The test uses a representative path that includes 'pp12h' and confirms that the function returns the integer value 12, indicating that it successfully parsed the lead time from the directory segment. This ensures that the regex-based extraction logic is functioning as intended for paths that follow the expected structure for forecast output files.

        Parameters:
            None

        Returns:
            None
        """
        assert extract_lead_time_hours_from_path("output/exp/ExtendedFC/2025061300/pp12h/foo.nc") == 12

    def test_no_match(self: "TestExtractLeadTime") -> None:
        """
        This test confirms that extract_lead_time_hours_from_path returns None when given a path that does not contain a 'ppNh' directory component. The function relies on this specific pattern to extract the lead time, so a path that lacks it should result in a None return value rather than raising an exception. This allows callers to gracefully handle paths that do not conform to the expected structure without crashing. The test uses a simple path with no recognizable 'ppNh' segment and confirms that the function returns None as expected. 

        Parameters:
            None

        Returns:
            None
        """
        assert extract_lead_time_hours_from_path("output/exp/foo.nc") is None


class TestNormalizeLongitudeBadTarget:
    """ Tests for normalize_longitude verifying that a ValueError is raised when an unrecognised target string is given. """

    def test_normalize_longitude_bad_target(self: "TestNormalizeLongitudeBadTarget") -> None:
        """
        This test ensures that normalize_longitude raises a ValueError when given an unrecognised target string that does not match either '0_360' or '-180_180'. Proper error handling is crucial to prevent silent failures and to allow callers to catch and handle invalid target specifications gracefully. The test constructs a DataArray with longitude coordinates and attempts to normalize it using an invalid target string, confirming that a ValueError is raised with an appropriate error message indicating the unknown longitude target. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.zeros((3, 5)),
            dims=["latitude", "longitude"],
            coords={"latitude": [0, 1, 2], "longitude": [0, 90, 180, 270, 350]},
        )
        with pytest.raises(ValueError, match="Unknown longitude target"):
            normalize_longitude(da, target="bad")


class TestParseFilenameMissingParts:
    """ Tests for parse_filename_metadata verifying behavior when required patterns are missing from filenames. """

    def test_parse_filename_missing_thresh(self: "TestParseFilenameMissingParts") -> None:
        """
        This test verifies that parse_fss_filename_metadata returns None when the threshold component is absent from the filename. The parser relies on specific tokens to extract metadata, so a filename that lacks the threshold information should result in a None return value rather than raising an exception. This allows callers to gracefully skip files that do not conform to the expected naming convention without crashing. The test uses a filename that includes the domain and window size but omits the threshold component and confirms that the function returns None as expected. 

        Parameters:
            None

        Returns:
            None
        """
        result = parse_fss_filename_metadata("modvx_metrics_type_neighborhood_global_1h_window15.nc")
        assert result is None

    def test_parse_filename_missing_window(self: "TestParseFilenameMissingParts") -> None:
        """
        This test verifies that parse_fss_filename_metadata returns None when the window size component is absent from the filename. Like the missing-threshold case, the parser requires both fields to be present and should return None when either is missing. This symmetric check ensures both required components are validated independently and that a filename with only the threshold present is also rejected.

        Parameters:
            None

        Returns:
            None
        """
        result = parse_fss_filename_metadata("modvx_metrics_type_neighborhood_global_1h_indep_thresh90p0percent.nc")
        assert result is None
