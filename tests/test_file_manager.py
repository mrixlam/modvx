"""Tests for modvx.file_manager — path construction and FIMERG helpers."""

import datetime

import pytest

from modvx.config import ModvxConfig
from modvx.file_manager import FileManager


@pytest.fixture
def fm() -> FileManager:
    """
    Construct a FileManager instance with a minimal ModvxConfig for path-construction tests. The base_dir is set to '/work' and a stub grid file path is supplied so that path-building logic can be exercised without requiring a real filesystem. Tests depending on this fixture can call FileManager methods directly without additional setup.

    Returns:
        FileManager: FileManager backed by a minimal hardcoded configuration.
    """
    return FileManager(ModvxConfig(base_dir="/work", mpas_grid_file="grid/x1.grid.nc"))


class TestForecastPath:
    """Tests for get_forecast_filepath verifying that MPAS diagnostic file paths are constructed correctly from valid-time and cycle arguments."""

    def test_mpas_diag_path(self, fm: FileManager) -> None:
        """
        Verify that get_forecast_filepath embeds the valid time in the correct MPAS diagnostic filename format. This test supplies a noon valid time for the 2024091700 forecast cycle and asserts the returned path contains both the formatted filename and the expected ExtendedFC subdirectory structure. Correct path construction is essential for the pipeline to locate forecast NetCDF files on disk without manual intervention.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 17, 12, 0, 0)
        path = fm.get_forecast_filepath(vt, "2024091700")
        assert "diag.2024-09-17_12.00.00.nc" in path
        assert "ExtendedFC/2024091700" in path

    def test_mpas_diag_midnight(self, fm: FileManager) -> None:
        """
        Confirm that midnight valid times are formatted correctly in the MPAS diagnostic filename with zero-padded hours. This edge case ensures the time component '00.00.00' appears in the constructed path, distinguishing midnight from any misformatted or omitted hour field. Midnight hours are common forecast lead times and must be handled without special-casing in upstream code.

        Returns:
            None
        """
        vt = datetime.datetime(2024, 9, 18, 0, 0, 0)
        path = fm.get_forecast_filepath(vt, "2024091700")
        assert "diag.2024-09-18_00.00.00.nc" in path


class TestObsHourIndex:
    """Tests for get_observation_hour_index verifying the mapping of valid times to zero-based hourly indices within FIMERG daily observation files."""

    def test_index_0(self) -> None:
        """
        Verify that a valid time of 01:00 maps to observation hour index 0 in the daily FIMERG file. FIMERG daily files store 24 hourly accumulation bands indexed from 0, where index 0 corresponds to the period ending at 01:00 UTC. This test guards against off-by-one errors in the index calculation that would cause incorrect hourly observation data to be loaded.

        Returns:
            None
        """
        # Time 01:00 → index 0
        assert FileManager.get_observation_hour_index(
            datetime.datetime(2025, 6, 14, 1, 0)
        ) == 0

    def test_midnight(self) -> None:
        """
        Confirm that a midnight valid time (00:00 UTC) maps to observation hour index 23. Midnight accumulations reference the last band of the preceding day's FIMERG file, since FIMERG daily files use a 01:00–00:00 UTC accumulation window. This test prevents the common error of associating midnight data with index 0 of the current day.

        Returns:
            None
        """
        # Midnight → index 23 (belongs to previous day's file)
        assert FileManager.get_observation_hour_index(
            datetime.datetime(2025, 6, 15, 0, 0)
        ) == 23

    def test_hour_12(self) -> None:
        """
        Verify that a valid time of 12:00 UTC maps to observation hour index 11 in the daily FIMERG file. This mid-day case confirms that the index formula correctly subtracts one from the hour value for all non-midnight times. Consistent index mapping across the full 01:00–23:00 range ensures observation accumulations align with their intended hourly periods during verification.

        Returns:
            None
        """
        assert FileManager.get_observation_hour_index(
            datetime.datetime(2025, 6, 14, 12, 0)
        ) == 11


class TestGroupObsTimes:
    """Tests for group_observation_times_by_date verifying correct grouping of valid times into FIMERG daily file buckets, including midnight crossover handling."""

    def test_single_day(self) -> None:
        """
        Verify that valid times within a single calendar day are grouped under the correct date key. This test generates six hourly times from 01:00 to 06:00 on 2025-06-14 and asserts they are placed in a single group keyed by '20250614'. Accurate single-day grouping is the baseline behavior required before testing the midnight boundary case.

        Returns:
            None
        """
        start = datetime.datetime(2025, 6, 14, 1)
        end = datetime.datetime(2025, 6, 14, 6)
        interval = datetime.timedelta(hours=1)
        groups = FileManager.group_observation_times_by_date(start, end, interval)
        assert "20250614" in groups
        assert len(groups["20250614"]) == 6  # hours 1-6

    def test_midnight_crossover(self) -> None:
        """
        Confirm that valid times spanning a midnight boundary are split correctly across two date-keyed groups. This test covers three consecutive hours — 23:00 on June 14, midnight, and 01:00 on June 15 — and asserts that midnight belongs to the prior day's group while 01:00 starts the next day's group. Correct midnight crossover handling is critical because FIMERG daily files use a 01:00–00:00 UTC accumulation window rather than a standard 00:00–23:00 calendar day.

        Returns:
            None
        """
        start = datetime.datetime(2025, 6, 14, 23)
        end = datetime.datetime(2025, 6, 15, 1)
        interval = datetime.timedelta(hours=1)
        groups = FileManager.group_observation_times_by_date(start, end, interval)
        # 23:00 → 20250614, 00:00 → 20250614, 01:00 → 20250615
        assert "20250614" in groups
        assert "20250615" in groups
        assert len(groups["20250614"]) == 2  # 23:00 and midnight
        assert len(groups["20250615"]) == 1  # 01:00
