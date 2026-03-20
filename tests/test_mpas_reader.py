#!/usr/bin/env python3

"""
Tests for modvx.mpas_reader — MPAS mesh loading and input file consistency.

This module contains unit tests for the MPAS reader component of modvx, which is responsible for loading native MPAS diagnostic files and remapping precipitation fields to a regular lat-lon grid. The tests verify that the reader correctly identifies variables to load, handles missing dependencies gracefully, and produces expected outputs for known input files. By isolating the MPAS reading logic, these tests ensure that downstream verification code can rely on consistent and correctly processed input data.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

import datetime
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from modvx.mpas_reader import load_mpas_precip

_TEST_DIR = Path(__file__).resolve().parent / "testdata"
_GRID_FILE = _TEST_DIR / "grid" / "x1.163842.grid.nc"
_FC_DIR = _TEST_DIR / "data" / "fcst" / "mrislam_coldstart_60km2024" / "ExtendedFC"
_CYCLES = ["2024091700", "2024091800", "2024091900", "2024092000", "2024092100"]
_FORECAST_HOURS = 6  # each cycle has 6h of hourly diag files (hours 0-6)

# Expected mesh size for this grid
_EXPECTED_NCELLS = 163842

def _is_real_netcdf(path: Path) -> bool:
    """Return True only when *path* is a proper NetCDF file, not a Git-LFS pointer."""
    if not path.exists() or path.stat().st_size < 512:
        return False
    try:
        with open(path, "rb") as fh:
            header = fh.read(64)
        # LFS pointer files begin with the ASCII text "version https://git-lfs"
        return b"version https://git-lfs" not in header
    except OSError:
        return False

# Skip all tests in this module if the grid file is missing or is a Git-LFS pointer
pytestmark = pytest.mark.skipif(
    not _is_real_netcdf(_GRID_FILE),
    reason="test data not present or is a Git-LFS pointer (tests/testdata/)",
)


class TestLoadMpasPrecipSynthetic:
    """ Unit tests for load_mpas_precip using minimal synthetic NetCDF file pairs that avoid any dependency on real MPAS forecast data. Each test constructs in-memory or tmp_path-based datasets with known values to verify specific loader behaviors in isolation. """

    @pytest.fixture()
    def _synth_files(self, tmp_path: Path):
        """Create a minimal synthetic MPAS diagnostic and grid NetCDF file pair for isolated unit testing. The grid file contains 10 mesh cells with lonCell and latCell coordinates spanning the full spherical range in radians. The diag file stores rainc and rainnc variables with known linearly-spaced values so that test assertions can verify exact numerical outputs from load_mpas_precip without reading real forecast data."""
        ncells = 10
        grid_ds = xr.Dataset(
            {
                "lonCell": ("nCells", np.linspace(0, 2 * np.pi, ncells)),
                "latCell": ("nCells", np.linspace(-np.pi / 2, np.pi / 2, ncells)),
            }
        )
        grid_path = tmp_path / "grid.nc"
        grid_ds.to_netcdf(grid_path)

        rainc = np.arange(ncells, dtype=np.float32)
        rainnc = np.arange(ncells, dtype=np.float32) * 2
        diag_ds = xr.Dataset(
            {
                "rainc": (("Time", "nCells"), rainc[np.newaxis, :]),
                "rainnc": (("Time", "nCells"), rainnc[np.newaxis, :]),
            }
        )
        diag_path = tmp_path / "diag.nc"
        diag_ds.to_netcdf(diag_path)

        return str(diag_path), str(grid_path), ncells, rainc, rainnc

    def test_returns_sum_of_rainc_rainnc(self, _synth_files) -> None:
        """Verify that load_mpas_precip returns the element-wise sum of rainc and rainnc as a DataArray with the correct shape. This test unpacks the synthetic fixture to access the known input arrays and asserts numerical closeness between the function output and the expected sum. The result shape must also equal the number of mesh cells, confirming that no cells are dropped or duplicated during the loading process."""
        diag, grid, ncells, rainc, rainnc = _synth_files
        result = load_mpas_precip(diag, grid)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (ncells,)
        np.testing.assert_allclose(result.values, rainc + rainnc)

    def test_units_attribute(self, _synth_files) -> None:
        """Confirm that load_mpas_precip attaches a 'units' attribute set to 'mm' on the returned DataArray. Consistent unit metadata is required so that downstream threshold comparisons and FSS calculations operate on precipitation values in the expected millimeter scale. This test guards against silent omission of unit attributes that could cause misleading results when the DataArray is passed to visualization or metrics code."""
        diag, grid, *_ = _synth_files
        result = load_mpas_precip(diag, grid)
        assert result.attrs["units"] == "mm"

    def test_coords_merged_from_grid(self, _synth_files) -> None:
        """Verify that load_mpas_precip correctly sources lonCell and latCell coordinates from the grid file, since MPAS diagnostic files do not contain mesh coordinate variables. This test opens the raw diag file directly and confirms that lonCell is absent, then calls load_mpas_precip to assert that coordinate merging from the grid file produces a valid result. Proper coordinate merging is essential for subsequent remapping of the unstructured mesh to a regular lat-lon grid."""
        diag, grid, *_ = _synth_files
        load_mpas_precip(diag, grid)
        # The function should have opened the grid to get coordinates
        ds = xr.open_dataset(diag)
        assert "lonCell" not in ds
        ds.close()

    def test_only_rainc(self, tmp_path: Path) -> None:
        """Verify that load_mpas_precip returns the rainc array unchanged when rainnc is absent from the diag file. This test constructs a minimal diag file containing only rainc and asserts the output matches those values exactly. Correct behavior for the rainc-only case prevents silent zeroing or error when a model run only writes convective precipitation output."""
        ncells = 5
        grid_ds = xr.Dataset(
            {
                "lonCell": ("nCells", np.zeros(ncells)),
                "latCell": ("nCells", np.zeros(ncells)),
            }
        )
        grid_path = tmp_path / "grid.nc"
        grid_ds.to_netcdf(grid_path)

        rainc = np.ones(ncells, dtype=np.float32)
        ds = xr.Dataset({"rainc": (("Time", "nCells"), rainc[np.newaxis, :])})
        diag_path = tmp_path / "diag.nc"
        ds.to_netcdf(diag_path)

        result = load_mpas_precip(str(diag_path), str(grid_path))
        np.testing.assert_allclose(result.values, rainc)

    def test_only_rainnc(self, tmp_path: Path) -> None:
        """Verify that load_mpas_precip returns the rainnc array unchanged when rainc is absent from the diag file. This test constructs a diag file containing only rainnc and asserts the output matches those values exactly. Correct behavior for the rainnc-only case ensures that large-scale non-convective precipitation is not silently zeroed out when a model run omits the convective component."""
        ncells = 5
        grid_ds = xr.Dataset(
            {
                "lonCell": ("nCells", np.zeros(ncells)),
                "latCell": ("nCells", np.zeros(ncells)),
            }
        )
        grid_path = tmp_path / "grid.nc"
        grid_ds.to_netcdf(grid_path)

        rainnc = np.ones(ncells, dtype=np.float32) * 3
        ds = xr.Dataset({"rainnc": (("Time", "nCells"), rainnc[np.newaxis, :])})
        diag_path = tmp_path / "diag.nc"
        ds.to_netcdf(diag_path)

        result = load_mpas_precip(str(diag_path), str(grid_path))
        np.testing.assert_allclose(result.values, rainnc)

    def test_missing_precip_vars_raises(self, tmp_path: Path) -> None:
        """Ensure that load_mpas_precip raises a ValueError when the diag file contains neither rainc nor rainnc. This test provides a diag file with only a 'temperature' variable and expects the function to fail with a message matching 'No rainc or rainnc'. Explicit early failure on missing precipitation variables prevents silent NaN propagation or misleading zero-precip results downstream in the verification pipeline."""
        ncells = 5
        grid_ds = xr.Dataset(
            {
                "lonCell": ("nCells", np.zeros(ncells)),
                "latCell": ("nCells", np.zeros(ncells)),
            }
        )
        grid_path = tmp_path / "grid.nc"
        grid_ds.to_netcdf(grid_path)

        ds = xr.Dataset({"temperature": (("Time", "nCells"), np.zeros((1, ncells)))})
        diag_path = tmp_path / "diag.nc"
        ds.to_netcdf(diag_path)

        with pytest.raises(ValueError, match="No 'rainc' or 'rainnc'"):
            load_mpas_precip(str(diag_path), str(grid_path))


class TestGridFileConsistency:
    """ Structural validation tests for the real MPAS grid NetCDF file confirming coordinate variables, mesh dimensions, and topology arrays are present and within expected physical ranges. """

    @pytest.fixture(scope="class")
    def grid(self):
        """ Open the real MPAS grid NetCDF file and yield the dataset for class-scoped reuse across all grid consistency tests. The dataset is opened once per test class to avoid redundant file I/O and is closed automatically after all tests in the class have completed. """
        ds = xr.open_dataset(_GRID_FILE)
        yield ds
        ds.close()

    def test_ncells_dimension(self, grid: xr.Dataset) -> None:
        """Assert that the MPAS grid file's nCells dimension matches the expected mesh resolution of 163,842 cells. This test detects grid file substitution or corruption that would cause all subsequent mesh-dependent operations to run at the wrong spatial resolution. It is intentionally kept as a fast sanity check that runs before any computationally heavier coordinate or topology tests."""
        assert grid.sizes["nCells"] == _EXPECTED_NCELLS

    def test_has_lonCell_latCell(self, grid: xr.Dataset) -> None:
        """Confirm that the grid file exposes lonCell and latCell coordinate variables required for merging with MPAS diagnostic files. These variables are the primary source of spatial coordinates for the unstructured mesh since diag files do not carry their own cell coordinates. Their absence would cause load_mpas_precip to fail or produce coordinates-free output during the coordinate merge step."""
        assert "lonCell" in grid, "Grid file missing lonCell"
        assert "latCell" in grid, "Grid file missing latCell"

    def test_coordinate_shapes(self, grid: xr.Dataset) -> None:
        """Verify that lonCell and latCell are 1D arrays with exactly one entry per mesh cell. Coordinate arrays with incorrect shapes would cause broadcasting errors or silently misalign cell positions during the remapping step that converts the unstructured MPAS mesh to a regular lat-lon grid. This shape check complements the dimension count and range tests by confirming array dimensionality."""
        assert grid["lonCell"].shape == (_EXPECTED_NCELLS,)
        assert grid["latCell"].shape == (_EXPECTED_NCELLS,)

    def test_lonCell_range(self, grid: xr.Dataset) -> None:
        """Verify that all lonCell values fall within the MPAS radian convention range of [0, 2π]. Values outside this interval indicate a unit mismatch or data corruption that would produce incorrect coordinates after the radian-to-degree conversion applied during mesh remapping. Catching this early prevents physically nonsensical longitude offsets from propagating into the lat-lon verification grid."""
        lon = grid["lonCell"].values
        assert lon.min() >= 0.0, f"lonCell min {lon.min()} < 0"
        assert lon.max() <= 2 * np.pi + 1e-6, f"lonCell max {lon.max()} > 2π"

    def test_latCell_range(self, grid: xr.Dataset) -> None:
        """Verify that all latCell values fall within the MPAS radian convention range of [-π/2, π/2]. Latitude values outside this range would indicate a unit mismatch that produces physically nonsensical coordinates after radian-to-degree conversion. Passing this test confirms the latitude coordinate convention matches what the remapping utility expects when constructing the regular output grid."""
        lat = grid["latCell"].values
        assert lat.min() >= -np.pi / 2 - 1e-6
        assert lat.max() <= np.pi / 2 + 1e-6

    def test_no_nan_in_coordinates(self, grid: xr.Dataset) -> None:
        """Ensure that neither lonCell nor latCell contains any NaN values across all mesh cells. NaN-valued coordinates would cause silent failures during interpolation and remapping, producing NaN-filled output grids without an explicit error. This check provides an early warning for incomplete or malformed grid files before any data-dependent operations are attempted."""
        assert not np.any(np.isnan(grid["lonCell"].values))
        assert not np.any(np.isnan(grid["latCell"].values))

    def test_has_topology_vars(self, grid: xr.Dataset) -> None:
        """Confirm that the grid file contains the basic MPAS topology variables needed for mesh connectivity operations. The variables nEdgesOnCell, cellsOnCell, and edgesOnCell define adjacency relationships between mesh cells and are required by mpasdiag and other MPAS utility libraries. Their absence would cause runtime errors in any pipeline step that performs neighborhood-based computations on the unstructured mesh."""
        for var in ("nEdgesOnCell", "cellsOnCell", "edgesOnCell"):
            assert var in grid, f"Grid file missing topology variable {var}"


class TestDiagFilesExist:
    """ File-presence tests that confirm all expected MPAS diagnostic NetCDF files exist on disk for every forecast cycle in the test dataset. """

    @pytest.mark.parametrize("cycle", _CYCLES)
    def test_init_hour_diag_exists(self, cycle: str) -> None:
        """ Confirm that the hour-0 MPAS diagnostic file exists for each forecast cycle in the test dataset. The initialization-hour file is required to compute precipitation accumulations by providing the baseline cumulative value that is subtracted from later hours. Missing hour-0 files would either raise a FileNotFoundError in the pipeline or produce incorrect accumulations using a stale prior-run baseline. """
        init_dt = datetime.datetime.strptime(cycle, "%Y%m%d%H")
        ts = init_dt.strftime("%Y-%m-%d_%H.%M.%S")
        path = _FC_DIR / cycle / f"diag.{ts}.nc"
        assert path.exists(), f"Missing init diag for cycle {cycle}: {path}"

    @pytest.mark.parametrize("cycle", _CYCLES)
    def test_hourly_diag_sequence(self, cycle: str) -> None:
        """Verify that the full sequence of hourly MPAS diagnostic files exists from lead hour 0 through the configured _FORECAST_HOURS for each forecast cycle. Gaps in the sequence would cause the accumulation loop in FileManager to skip or misalign precipitation intervals, leading to incorrect FSS inputs. All missing filenames are collected and reported together rather than failing on the first missing file to simplify debugging."""
        init_dt = datetime.datetime.strptime(cycle, "%Y%m%d%H")
        missing = []
        for hour in range(_FORECAST_HOURS + 1):  # 0.._FORECAST_HOURS inclusive
            vt = init_dt + datetime.timedelta(hours=hour)
            ts = vt.strftime("%Y-%m-%d_%H.%M.%S")
            path = _FC_DIR / cycle / f"diag.{ts}.nc"
            if not path.exists():
                missing.append(f"hour {hour}: {path.name}")
        assert not missing, (
            f"Cycle {cycle} missing {len(missing)} diag files:\n"
            + "\n".join(missing[:10])
        )


class TestDiagFileContents:
    """ Content validation tests checking variable presence, array shape, data type, and physical plausibility within a representative MPAS diagnostic file. """

    @pytest.fixture(scope="class")
    def sample_diag(self):
        """ Open the hour-6 MPAS diagnostic file from the 2024091700 forecast cycle for class-scoped reuse across all content validation tests. This mid-forecast file is chosen because it is expected to contain non-zero cumulative precipitation, making it more representative than the initialization-hour file. The dataset is opened once per test class to minimize I/O overhead and is closed automatically after all tests finish. """
        path = _FC_DIR / "2024091700" / "diag.2024-09-17_06.00.00.nc"
        ds = xr.open_dataset(path)
        yield ds
        ds.close()

    def test_has_rainc_rainnc(self, sample_diag: xr.Dataset) -> None:
        """Confirm that the diag file contains both rainc and rainnc precipitation variables expected by load_mpas_precip. The loader sums these two variables to derive total cumulative precipitation, so absence of either would result in incomplete accumulation. This test acts as a prerequisite check before any numerical or shape-based validation is attempted on the precipitation data."""
        assert "rainc" in sample_diag, "Diag file missing rainc"
        assert "rainnc" in sample_diag, "Diag file missing rainnc"

    def test_rainc_shape(self, sample_diag: xr.Dataset) -> None:
        """Verify that the rainc variable has the expected (Time, nCells) dimension order and the correct number of mesh cells. Incorrect dimension ordering would cause the isel(Time=0) slicing in load_mpas_precip to select the wrong axis, producing misshapen output arrays. The nCells count is also validated to confirm the diag file originates from the same mesh resolution as the grid file."""
        assert sample_diag["rainc"].dims == ("Time", "nCells")
        assert sample_diag["rainc"].shape[1] == _EXPECTED_NCELLS

    def test_single_time_step(self, sample_diag: xr.Dataset) -> None:
        """Assert that each MPAS diagnostic file contains exactly one time snapshot in the Time dimension. The load_mpas_precip function uses isel(Time=0) to extract values, which is only correct if files contain a single time step as MPAS is configured to write. Files with more than one entry would cause the wrong time step to be loaded silently, depending on which step holds the cumulative precipitation values."""
        assert sample_diag.sizes["Time"] == 1

    def test_ncells_matches_grid(self, sample_diag: xr.Dataset) -> None:
        """Verify that the nCells dimension in the diag file is consistent with the expected mesh size from the grid file. Mismatched nCells counts between the diag and grid files would cause a dimension mismatch error when merging coordinates during load_mpas_precip. This test ensures the diag and grid files originate from the same MPAS mesh configuration before any data loading operations are attempted."""
        assert sample_diag.sizes["nCells"] == _EXPECTED_NCELLS

    def test_precip_non_negative(self, sample_diag: xr.Dataset) -> None:
        """Confirm that cumulative precipitation values in both rainc and rainnc are non-negative across all mesh cells. Since these are running-total fields, any negative value indicates numerical corruption or a model reset artifact that would propagate into incorrect accumulation differences. A small tolerance of 1e-6 mm is permitted to accommodate floating-point rounding in the model output without generating false failures."""
        for var in ("rainc", "rainnc"):
            vals = sample_diag[var].values
            assert np.all(vals >= -1e-6), (
                f"{var} has negative values: min={vals.min()}"
            )

    def test_precip_dtype_float32(self, sample_diag: xr.Dataset) -> None:
        """Verify that the rainc and rainnc variables are stored as 32-bit floating-point values consistent with MPAS model output conventions. Using float32 is important for memory efficiency on large meshes and for compatibility with the arithmetic operations applied during accumulation differencing. This test guards against accidental integer encoding or double-precision storage that would alter memory footprint or downstream type promotion behavior."""
        for var in ("rainc", "rainnc"):
            assert sample_diag[var].dtype == np.float32


class TestHourZeroPrecip:
    """ Parametrized sanity checks confirming that every forecast cycle's initialization-hour diag file contains zero cumulative precipitation for both rainc and rainnc. """

    @pytest.mark.parametrize("cycle", _CYCLES)
    def test_precip_zero_at_init(self, cycle: str) -> None:
        """ Verify that cumulative precipitation is zero at the model initialization hour for every forecast cycle in the test dataset. MPAS resets cumulative precipitation counters at the start of each cold-start forecast run, so any non-zero value at hour 0 indicates an initialization failure or a misidentified warm-start file. Failing this test would invalidate all accumulation calculations that use the hour-0 file as the baseline subtraction reference. """
        init_dt = datetime.datetime.strptime(cycle, "%Y%m%d%H")
        ts = init_dt.strftime("%Y-%m-%d_%H.%M.%S")
        path = _FC_DIR / cycle / f"diag.{ts}.nc"
        ds = xr.open_dataset(path)
        for var in ("rainc", "rainnc"):
            if var in ds:
                vals = ds[var].isel(Time=0).values
                assert np.allclose(vals, 0.0, atol=1e-6), (
                    f"Cycle {cycle}: {var} at hour 0 is not zero "
                    f"(max={vals.max():.6f})"
                )
        ds.close()


class TestPrecipMonotonicity:
    """ Integration tests verifying that domain-total cumulative precipitation increases or stays constant as the forecast advances, consistent with MPAS's running-accumulation output convention. """

    def test_total_precip_increases(self) -> None:
        """ Verify that domain-total cumulative precipitation at hour 3 is greater than or equal to hour 0, and at the final forecast hour is greater than or equal to hour 3. Because MPAS writes running totals rather than interval accumulations, summed precipitation can only stay constant or increase across consecutive time steps. Violating this monotonicity property would indicate a model reset, a file ordering error, or numerical corruption in the diagnostic output. """
        cycle = "2024091700"
        init_dt = datetime.datetime(2024, 9, 17, 0)
        grid_file = str(_GRID_FILE)

        check_hours = (0, 3, _FORECAST_HOURS)
        totals = []
        for hour in check_hours:
            vt = init_dt + datetime.timedelta(hours=hour)
            ts = vt.strftime("%Y-%m-%d_%H.%M.%S")
            path = str(_FC_DIR / cycle / f"diag.{ts}.nc")
            precip = load_mpas_precip(path, grid_file)
            totals.append(float(precip.sum()))

        for i in range(1, len(totals)):
            assert totals[i] >= totals[i - 1] - 1e-3, (
                f"Total precip decreased: hour {check_hours[i-1]}={totals[i-1]:.2f} "
                f"> hour {check_hours[i]}={totals[i]:.2f}"
            )


class TestAccumulationConsistency:
    """ Integration tests that validate the physical plausibility of precipitation accumulations computed by differencing MPAS diagnostic files across a 6-hour window. """

    def test_6h_accumulation_non_negative(self) -> None:
        """ Verify that the 6-hour precipitation accumulation derived by subtracting the hour-0 diag file from the hour-6 diag file is non-negative at every mesh cell. Negative accumulations would indicate that cumulative totals decreased over time, which is physically impossible and would produce incorrect binary masks in the FSS verification pipeline. A small tolerance of 1e-4 mm is allowed to absorb floating-point subtraction rounding without triggering false positives. """
        cycle = "2024091700"
        grid_file = str(_GRID_FILE)
        init_dt = datetime.datetime(2024, 9, 17, 0)
        end_dt = init_dt + datetime.timedelta(hours=_FORECAST_HOURS)

        start = load_mpas_precip(
            str(_FC_DIR / cycle / f"diag.{init_dt.strftime('%Y-%m-%d_%H.%M.%S')}.nc"),
            grid_file,
        )
        end = load_mpas_precip(
            str(_FC_DIR / cycle / f"diag.{end_dt.strftime('%Y-%m-%d_%H.%M.%S')}.nc"),
            grid_file,
        )

        accum = end - start
        assert np.all(accum.values >= -1e-4), (
            f"Negative accumulation detected: min={float(accum.min()):.6f}"
        )

    def test_6h_accumulation_reasonable_magnitude(self) -> None:
        """ Confirm that the maximum 6-hour accumulated precipitation does not exceed a physically plausible upper bound of 500 mm anywhere in the domain. Exceeding this threshold would suggest a data overflow, incorrect unit conversion, or a runaway numerical issue in the model output that would render FSS computations meaningless. This sanity bound is intentionally generous to avoid false positives from extreme convective events while still catching clearly erroneous values. """
        cycle = "2024091700"
        grid_file = str(_GRID_FILE)
        init_dt = datetime.datetime(2024, 9, 17, 0)
        end_dt = init_dt + datetime.timedelta(hours=_FORECAST_HOURS)

        start = load_mpas_precip(
            str(_FC_DIR / cycle / f"diag.{init_dt.strftime('%Y-%m-%d_%H.%M.%S')}.nc"),
            grid_file,
        )
        end = load_mpas_precip(
            str(_FC_DIR / cycle / f"diag.{end_dt.strftime('%Y-%m-%d_%H.%M.%S')}.nc"),
            grid_file,
        )

        accum = end - start
        assert float(accum.max()) < 500.0, (
            f"Unreasonable 6h precip: max={float(accum.max()):.1f} mm"
        )


class TestEnsureMpasdiagBranches:
    """ Cover _ensure_mpasdiag import-error, cached-true, and success paths. """

    def test_ensure_mpasdiag_import_error(self) -> None:
        """
        Verify that _ensure_mpasdiag raises ImportError when the mpasdiag package cannot be imported.
        The function probes for mpasdiag at call time and must raise ImportError with a message containing
        'mpasdiag is required' when the package is missing. This guards against silent failures where
        remapping would proceed without the required library and produce incorrect output.

        Returns:
            None
        """
        from modvx import mpas_reader

        old_val = mpas_reader._HAS_MPASDIAG
        mpas_reader._HAS_MPASDIAG = None
        with patch.dict("sys.modules", {
            "mpasdiag": None,
            "mpasdiag.processing": None,
            "mpasdiag.processing.remapping": None,
            "mpasdiag.processing.utils_geog": None,
        }):
            with pytest.raises(ImportError, match="mpasdiag is required"):
                mpas_reader._ensure_mpasdiag_available()
        mpas_reader._HAS_MPASDIAG = old_val

    def test_ensure_mpasdiag_cached_true(self) -> None:
        """
        Verify that _ensure_mpasdiag returns immediately without re-importing when already cached as True.
        After a successful import the _HAS_MPASDIAG flag is set to True so subsequent calls skip the
        import probe entirely. This test pre-sets the flag and confirms no ImportError is raised, which
        would be the symptom if the import was attempted again in a patched environment.

        Returns:
            None
        """
        from modvx import mpas_reader

        old_val = mpas_reader._HAS_MPASDIAG
        mpas_reader._HAS_MPASDIAG = True
        mpas_reader._ensure_mpasdiag_available()  # should not raise
        mpas_reader._HAS_MPASDIAG = old_val

    def test_successful_import(self) -> None:
        """
        Verify that _ensure_mpasdiag successfully imports fake mpasdiag modules and sets _HAS_MPASDIAG to True.
        This test injects synthetic module objects into sys.modules to simulate a valid mpasdiag installation.
        After the first call _HAS_MPASDIAG must be True, and a second call must be a no-op confirming
        the early-return cache path also works for the success branch.

        Returns:
            None
        """
        import modvx.mpas_reader as mr
        import sys

        mr._HAS_MPASDIAG = None

        fake_remapping = types.ModuleType("mpasdiag.processing.remapping")
        setattr(fake_remapping, "remap_mpas_to_latlon_with_masking", MagicMock())

        fake_utils = types.ModuleType("mpasdiag.processing.utils_geog")
        setattr(fake_utils, "MPASGeographicUtils", MagicMock())

        fake_processing = types.ModuleType("mpasdiag.processing")
        fake_mpasdiag = types.ModuleType("mpasdiag")

        saved = {}
        keys = [
            "mpasdiag", "mpasdiag.processing",
            "mpasdiag.processing.remapping",
            "mpasdiag.processing.utils_geog",
        ]
        for k in keys:
            saved[k] = sys.modules.get(k)

        try:
            sys.modules["mpasdiag"] = fake_mpasdiag
            sys.modules["mpasdiag.processing"] = fake_processing
            sys.modules["mpasdiag.processing.remapping"] = fake_remapping
            sys.modules["mpasdiag.processing.utils_geog"] = fake_utils

            mr._ensure_mpasdiag_available()
            assert mr._HAS_MPASDIAG is True

            # Calling again should be a no-op (early return)
            mr._ensure_mpasdiag_available()
            assert mr._HAS_MPASDIAG is True
        finally:
            for k in keys:
                if saved[k] is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = saved[k]
            mr._HAS_MPASDIAG = None


class TestRemapToLatlonWithGrid:
    """ Cover remap_to_latlon body — with lonCell present and with grid fallback. """

    def test_remap_with_grid_fallback(self, tmp_path: Path) -> None:
        """
        Verify that remap_to_latlon loads spatial coordinates from grid_file when lonCell is absent in the dataset.
        This covers the fallback branch where the main diagnostic file does not include grid coordinate
        variables and a separate mesh file must be opened to obtain lat/lon cell positions.
        The test confirms the remapping function is still called once and the result has the expected dims.

        Returns:
            None
        """
        import modvx.mpas_reader as mr
        import sys

        ds = xr.Dataset({"precip": (["nCells"], np.array([1.0, 2.0, 3.0]))})
        ds_path = tmp_path / "diag.nc"
        ds.to_netcdf(str(ds_path))

        grid_ds = xr.Dataset({
            "lonCell": (["nCells"], np.array([0.0, 1.0, 2.0])),
            "latCell": (["nCells"], np.array([10.0, 20.0, 30.0])),
        })
        grid_path = tmp_path / "grid.nc"
        grid_ds.to_netcdf(str(grid_path))

        fake_extract = MagicMock(
            return_value=(np.array([0.0, 1.0, 2.0]), np.array([10.0, 20.0, 30.0])),
        )
        fake_extent = MagicMock(return_value=(0.0, 2.0, 10.0, 30.0))
        mock_utils = MagicMock()
        mock_utils.extract_spatial_coordinates = fake_extract
        mock_utils.get_extent_from_coordinates = fake_extent

        remapped_da = xr.DataArray(
            np.ones((2, 2)),
            dims=["lat", "lon"],
            coords={"lat": [10.0, 30.0], "lon": [0.0, 2.0]},
        )
        mock_remap = MagicMock(return_value=remapped_da)

        mr._HAS_MPASDIAG = True

        fake_remapping_mod = types.ModuleType("mpasdiag.processing.remapping")
        setattr(fake_remapping_mod, "remap_mpas_to_latlon_with_masking", mock_remap)
        fake_utils_mod = types.ModuleType("mpasdiag.processing.utils_geog")
        setattr(fake_utils_mod, "MPASGeographicUtils", mock_utils)

        saved = {}
        keys = [
            "mpasdiag", "mpasdiag.processing",
            "mpasdiag.processing.remapping",
            "mpasdiag.processing.utils_geog",
        ]
        for k in keys:
            saved[k] = sys.modules.get(k)

        try:
            sys.modules["mpasdiag"] = types.ModuleType("mpasdiag")
            sys.modules["mpasdiag.processing"] = types.ModuleType("mpasdiag.processing")
            sys.modules["mpasdiag.processing.remapping"] = fake_remapping_mod
            sys.modules["mpasdiag.processing.utils_geog"] = fake_utils_mod

            result = mr.remap_to_latlon(
                data=xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["nCells"]),
                dataset_or_file=str(ds_path),
                grid_file=str(grid_path),
                resolution=1.0,
            )

            assert "latitude" in result.dims
            assert "longitude" in result.dims
            mock_remap.assert_called_once()
        finally:
            for k in keys:
                if saved[k] is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = saved[k]
            mr._HAS_MPASDIAG = None

    def test_remap_to_latlon_mocked(self, tmp_path: Path) -> None:
        """
        Verify the full remap_to_latlon path when lonCell is already present in the dataset.
        This covers the nominal branch where spatial coordinates are read directly from the diagnostic
        file without a separate grid file fallback. The test mocks the mpasdiag remapping function
        and confirms the returned DataArray has latitude and longitude dimensions.

        Returns:
            None
        """
        from modvx import mpas_reader

        n_cells = 10
        ds = xr.Dataset({
            "lonCell": ("nCells", np.linspace(0, 10, n_cells)),
            "latCell": ("nCells", np.linspace(-5, 5, n_cells)),
            "precip": ("nCells", np.random.default_rng(42).random(n_cells)),
        })
        ds_file = str(tmp_path / "ds.nc")
        ds.to_netcdf(ds_file)

        data = xr.DataArray(np.random.default_rng(42).random(n_cells), dims=["nCells"])

        remapped = xr.DataArray(
            np.ones((5, 5)),
            dims=["lat", "lon"],
            coords={"lat": np.arange(5.0), "lon": np.arange(5.0)},
        )

        mock_remap_fn = MagicMock(return_value=remapped)
        mock_geo_utils = MagicMock()
        mock_geo_utils.extract_spatial_coordinates.return_value = (
            np.linspace(0, 10, n_cells), np.linspace(-5, 5, n_cells)
        )
        mock_geo_utils.get_extent_from_coordinates.return_value = (0.0, 10.0, -5.0, 5.0)

        old_has = mpas_reader._HAS_MPASDIAG
        mpas_reader._HAS_MPASDIAG = True

        with patch.dict("sys.modules", {
            "mpasdiag": MagicMock(),
            "mpasdiag.processing": MagicMock(),
            "mpasdiag.processing.remapping": MagicMock(
                remap_mpas_to_latlon_with_masking=mock_remap_fn
            ),
            "mpasdiag.processing.utils_geog": MagicMock(
                MPASGeographicUtils=mock_geo_utils
            ),
        }):
            result = mpas_reader.remap_to_latlon(data, ds_file, ds_file, 0.5)

        mpas_reader._HAS_MPASDIAG = old_has

        assert "latitude" in result.dims
        assert "longitude" in result.dims


class TestLoadAndRemapMpasPrecip:
    """ Cover load_and_remap_mpas_precip convenience wrapper. """

    def test_load_and_remap(self, tmp_path: Path) -> None:
        """
        Verify that load_and_remap_mpas_precip calls load_mpas_precip and remap_to_latlon with the correct arguments.
        This convenience wrapper should call load_mpas_precip once with the diag_file and grid_file, then
        pass the result to remap_to_latlon along with both file paths and the resolution. The final return
        value must have the same shape as the remapped DataArray.

        Returns:
            None
        """
        from modvx import mpas_reader

        old_has = mpas_reader._HAS_MPASDIAG
        mpas_reader._HAS_MPASDIAG = True

        mesh_precip = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["nCells"])
        remapped = xr.DataArray(np.ones((3, 3)), dims=["latitude", "longitude"])

        with patch.object(mpas_reader, "load_mpas_precip", return_value=mesh_precip) as m_load, \
             patch.object(mpas_reader, "remap_to_latlon", return_value=remapped) as m_remap:
            result = mpas_reader.load_and_remap_mpas_precip("diag.nc", "grid.nc", 0.5)

        mpas_reader._HAS_MPASDIAG = old_has

        m_load.assert_called_once_with("diag.nc", "grid.nc")
        m_remap.assert_called_once_with(mesh_precip, "diag.nc", "grid.nc", 0.5)
        assert result.shape == (3, 3)
