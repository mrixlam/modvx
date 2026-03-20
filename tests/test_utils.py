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
    """ Tests for parse_datetime_string verifying correct parsing of both compact canonical and ISO 8601 datetime formats, as well as error handling for invalid strings. These tests ensure that the function can robustly handle the primary timestamp formats used in modvx configuration files and metadata, preventing silent misconfigurations due to parsing errors and ensuring compatibility with a range of datetime string inputs that may appear in different contexts within the pipeline. """

    def test_canonical(self) -> None:
        """
        Verify that parse_datetime correctly parses the compact canonical format 'YYYYMMDDThh' used throughout modvx configuration files. This is the primary timestamp format for initial_cycle_start and final_cycle_start fields in YAML configs, so parsing errors here would silently misconfigure the entire forecast cycle range. The test checks both the date and hour components to confirm no truncation or offset occurs.
        """
        assert parse_datetime_string("20250613T00") == datetime.datetime(2025, 6, 13, 0, 0)

    def test_iso(self) -> None:
        """
        Confirm that parse_datetime also accepts ISO 8601 extended format with full date, time, and second components. Supporting this format ensures compatibility with datetime strings that may appear in NetCDF metadata, log output, or user-supplied overrides without requiring a separate parsing code path. The parsed result is compared against a known datetime object to verify all components are correctly extracted.
        """
        assert parse_datetime_string("2024-09-23T06:00:00") == datetime.datetime(2024, 9, 23, 6, 0)

    def test_invalid_raises(self) -> None:
        """
        Ensure that parse_datetime raises a ValueError when given a string that does not match any supported datetime format. Explicit failure on unrecognized input prevents invalid datetime objects from propagating into forecast cycle generators and producing nonsensical date ranges. This test uses an obviously malformed string to confirm that the format detection logic does not silently fall through to a default value.
        """
        with pytest.raises(ValueError):
            parse_datetime_string("not-a-date")


class TestFormatThreshold:
    """ Tests for format_threshold_for_filename verifying correct conversion of float threshold values to filename-safe strings. These tests ensure that both integer-like and decimal thresholds are formatted consistently, preventing file naming collisions and ensuring compatibility with downstream loading code. The 'p' separator is used to replace the decimal point to avoid filesystem issues, and the formatting must be precise to ensure that the original float value can be reconstructed from the filename when reading saved results. """

    def test_integer_like(self) -> None:
        """
        Verify that a float with a zero decimal part is formatted as 'NNp0' without any trailing precision beyond the tenths place. This format is used to construct unique output filenames for each FSS threshold, so incorrect formatting would cause file naming collisions or mismatches when reading previously saved results. The 'p' separator replaces the decimal point to avoid filesystem path issues on all platforms.
        """
        assert format_threshold_for_filename(90.0) == "90p0"

    def test_decimal(self) -> None:
        """
        Confirm that a float with a non-zero decimal part is formatted by replacing the decimal point with 'p' to produce a filename-safe string. This covers the commonly used 97.5th percentile threshold, which must appear as '97p5' in output filenames so that downstream loading code can reconstruct the original float value by splitting on 'p'. Incorrect decimal handling here would cause percentile-specific FSS result files to be written to or read from wrong paths.
        """
        assert format_threshold_for_filename(97.5) == "97p5"


class TestGenerators:
    """ Tests for generate_valid_times and generate_forecast_cycles verifying sequence length, endpoint inclusion, and correct step-based iteration. These tests ensure that the generators produce the expected number of timestamps or cycles, correctly handle inclusive and exclusive endpoints, and iterate with the specified step size, preventing off-by-one errors and ensuring accurate coverage of the forecast period. """

    def test_valid_times(self) -> None:
        """
        Verify that generate_valid_times produces the correct number of timestamps and that the first and last values match the expected start and penultimate times. The generator uses a half-open convention where the start is included but the end is excluded, so a 6-hour range with a 2-hour step should yield exactly 3 elements stopping at hour 4. Confirming the exclusion of the endpoint prevents off-by-one errors that would cause extra forecast-observation pairs to be evaluated at or beyond the configured forecast length.
        """
        start = datetime.datetime(2025, 1, 1, 0)
        end = datetime.datetime(2025, 1, 1, 6)
        step = datetime.timedelta(hours=2)
        result = list(iterate_valid_times(start, end, step))
        assert len(result) == 3
        assert result[0] == start
        assert result[-1] == datetime.datetime(2025, 1, 1, 4)

    def test_forecast_cycles_inclusive(self) -> None:
        """
        Confirm that generate_forecast_cycles produces a sequence that includes both the start and end cycle datetimes. Unlike generate_valid_times, the cycle generator uses an inclusive convention to ensure the final configured cycle is always processed. A 3-day window with a 24-hour step must yield exactly 3 cycles, and any off-by-one in the endpoint condition would silently drop the last cycle from the verification run.
        """
        start = datetime.datetime(2025, 1, 1, 0)
        end = datetime.datetime(2025, 1, 3, 0)
        step = datetime.timedelta(hours=24)
        result = list(iterate_forecast_cycle_starts(start, end, step))
        assert len(result) == 3  # day 1, 2, 3


class TestNormalizeLongitude:
    """ Tests for normalize_longitude verifying correct remapping of longitude coordinates to the specified target convention. These tests ensure that negative longitudes are correctly converted to the [0, 360) range when requested, and that longitudes above 180 degrees are remapped to the [-180, 180) range when that convention is requested. Proper normalization is critical for spatial operations that require consistent longitude conventions across datasets, such as regridding or mask application, and these tests confirm that the function handles both common remapping scenarios correctly without altering non-target coordinates. """

    def test_negative_to_0_360(self) -> None:
        """
        Verify that normalize_longitude converts negative longitude coordinates to the [0, 360) range when the '0_360' convention is requested. Observation datasets such as FIMERG use the [0, 360) convention while some MPAS outputs may use [-180, 180), so consistent normalization is required before any spatial operations can be applied across both grids. This test confirms that all output longitude values are non-negative and at most 360 after conversion.
        """
        da = xr.DataArray(
            np.ones((2, 4)),
            dims=["latitude", "longitude"],
            coords={"latitude": [0, 1], "longitude": [-180, -90, 0, 90]},
        )
        result = normalize_longitude(da, "0_360")
        assert float(result.longitude.min()) >= 0
        assert float(result.longitude.max()) <= 360

    def test_positive_to_neg180(self) -> None:
        """
        Confirm that normalize_longitude converts longitude coordinates above 180 degrees to the [-180, 180) range when the '-180_180' convention is requested. Verification mask files and some regional domain specifications use the [-180, 180) convention, so remapping is needed before performing clipping or mask application. This test checks that a 270-degree input is correctly remapped to a negative value within the valid range.
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
    """ Tests for standardize_coords verifying dimension renaming from short-form to canonical long-form coordinate names and no-op behavior when names are already correct. These tests ensure that all downstream pipeline components can rely on consistent dimension names, preventing errors in indexing, alignment, and aggregation operations. """

    def test_rename_lat_lon(self) -> None:
        """
        Verify that standardize_coords renames 'lat' and 'lon' dimensions to 'latitude' and 'longitude'. Canonical long-form dimension names are required by all downstream pipeline components that index or coordinate-align arrays using named dimensions. This test constructs a bare DataArray without coordinate values to confirm renaming operates on dimension names independent of coordinate label presence.
        """
        da = xr.DataArray(np.ones((3, 3)), dims=["lat", "lon"])
        result = standardize_coords(da)
        assert "latitude" in result.dims
        assert "longitude" in result.dims

    def test_noop_when_standard(self) -> None:
        """
        Confirm that standardize_coords returns a DataArray with unchanged dimension names when they are already in the canonical 'latitude'/'longitude' format. This no-op behavior ensures the function is safe to call unconditionally on any input without risk of double-renaming or raising a KeyError when standard names are already present. The test verifies 'latitude' remains in dims after the call to confirm the function does not inadvertently rename or drop correct dimension names.
        """
        da = xr.DataArray(np.ones((3, 3)), dims=["latitude", "longitude"])
        result = standardize_coords(da)
        assert "latitude" in result.dims


class TestParseFilenameMetadata:
    """ Tests for parse_filename_metadata verifying regex-based extraction of domain, threshold, and window tokens from FSS output filenames. These tests ensure that the function can correctly interpret the structured naming convention used for FSS output files, allowing downstream code to reliably extract metadata for plotting, filtering, and analysis. """

    def test_standard(self) -> None:
        """
        Verify that parse_filename_metadata correctly extracts the domain, threshold, and window fields from a standard FSS output filename. The returned dictionary must contain all three keys with values matching the literal tokens encoded in the filename, since these are used to reconstruct plot labels and filter results during post-processing. This test uses a 'GLOBAL' domain with an integer-valued threshold to cover the most common output filename pattern.
        """
        m = parse_fss_filename_metadata(
            "modvx_metrics_type_neighborhood_global_12h_indep_thresh90p0percent_window3.nc"
        )
        assert m == {"domain": "global", "thresh": "90.0", "window": "3"}

    def test_decimal_thresh(self) -> None:
        """
        Confirm that parse_filename_metadata correctly extracts a decimal threshold value such as 97.5 from the 'p'-separated filename encoding. The extracted threshold string must be the reconstructed decimal form '97.5' rather than the raw 'p'-encoded form '97p5', so that it can be used directly in plot labels and numeric comparisons. This test covers the tropical 97.5th percentile threshold, which is among the most frequently used non-integer thresholds in the verification configuration.
        """
        m = parse_fss_filename_metadata(
            "modvx_metrics_type_neighborhood_tropics_6h_indep_thresh97p5percent_window11.nc"
        )
        assert m is not None
        assert m["thresh"] == "97.5"

    def test_bad_filename(self) -> None:
        """
        Ensure that parse_filename_metadata returns None when the filename does not match the expected FSS output naming convention. Silent None returns allow callers to filter out unrelated files in a directory scan without raising exceptions or crashing the post-processing pipeline. This test uses a generic filename with no FSS-specific tokens to confirm the function degrades gracefully on unexpected input.
        """
        assert parse_fss_filename_metadata("random_file.nc") is None


class TestExtractLeadTime:
    """ Tests for extract_lead_time_hours_from_path verifying correct regex extraction of lead time hours from output paths containing 'ppNh' directory components. These tests ensure that the function can reliably parse lead time information embedded in directory names, which is critical for annotating FSS results by forecast hour and organizing output files without requiring metadata to be stored in the files themselves. The function must return an integer representing the lead time in hours when the expected pattern is present, and return None when no such pattern exists, allowing downstream code to handle both cases appropriately. """

    def test_pp12h(self) -> None:
        """
        Verify that extract_lead_time_hours correctly parses a 12-hour lead time from an output path containing the 'pp12h' directory component. Lead time information embedded in directory names is used to annotate FSS results by forecast hour and to organize output files without requiring metadata to be stored in the files themselves. The function must return the integer 12 rather than the string '12' or None for this path pattern.
        """
        assert extract_lead_time_hours_from_path("output/exp/ExtendedFC/2025061300/pp12h/foo.nc") == 12

    def test_no_match(self) -> None:
        """
        Confirm that extract_lead_time_hours returns None when the path contains no 'ppNh' lead-time directory segment. Returning None for non-matching paths allows callers to skip files that do not follow the expected directory structure without raising exceptions. This test uses a flat two-component path to confirm the regex does not produce false positives on paths that have no lead-time encoding.
        """
        assert extract_lead_time_hours_from_path("output/exp/foo.nc") is None


class TestNormalizeLongitudeBadTarget:
    """ Tests for normalize_longitude verifying that a ValueError is raised when an unrecognised target string is given. The function supports '0to360' and '-180to180' longitude conventions; any other value should immediately raise a ValueError with a message containing 'Unknown longitude target'. This test protects callers from silent wrong-convention remapping when a typo or unknown convention name is passed. """

    def test_normalize_longitude_bad_target(self) -> None:
        """
        Confirm that normalize_longitude raises ValueError when an unrecognised target string is given. The function supports '0to360' and '-180to180' longitude conventions; any other value should immediately raise a ValueError with a message containing 'Unknown longitude target'. This test protects callers from silent wrong-convention remapping when a typo or unknown convention name is passed.

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
    """ Tests for parse_filename_metadata verifying behavior when required patterns are missing from filenames. These tests ensure that the function gracefully returns None when either the threshold or window size component is absent, allowing callers to skip unrecognised files without raising exceptions. """

    def test_parse_filename_missing_thresh(self) -> None:
        """
        Verify that parse_filename_metadata returns None when the threshold component is absent from the filename. The parser requires both a threshold and a window size to successfully extract metadata from an FSS output filename. A filename missing the threshold pattern should return None rather than raising an exception, allowing callers to skip unrecognised files gracefully.

        Returns:
            None
        """
        result = parse_fss_filename_metadata("modvx_metrics_type_neighborhood_global_1h_window15.nc")
        assert result is None

    def test_parse_filename_missing_window(self) -> None:
        """
        Verify that parse_filename_metadata returns None when the window size component is absent from the filename. Like the missing-threshold case, the parser requires both fields to be present and should return None when either is missing. This symmetric check ensures both required components are validated independently and that a filename with only the threshold present is also rejected.

        Returns:
            None
        """
        result = parse_fss_filename_metadata("modvx_metrics_type_neighborhood_global_1h_indep_thresh90p0percent.nc")
        assert result is None
