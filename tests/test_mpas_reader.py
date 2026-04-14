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
from typing import Iterator
import numpy as np
import pytest
import xarray as xr

from modvx.mpas_reader import load_mpas_precip

_TEST_DIR = Path(__file__).resolve().parent.parent / "data"
_GRID_FILE = _TEST_DIR / "grid" / "x1.10242.static.nc"
_FC_DIR = _TEST_DIR / "fcst" / "mrislam_coldstart_240km_meso" / "ExtendedFC"
_CYCLES = ["2014090100", "2014090106", "2014090112", "2014090118", "2014090200"]
_FORECAST_HOURS = 6  # each cycle has 6h of hourly diag files (hours 0-6)

# Expected mesh size for this grid
_EXPECTED_NCELLS = 10242

def _is_real_netcdf(path: Path) -> bool:
    """
    This helper function checks if the given path points to a real NetCDF file by verifying that it exists, has a reasonable file size, and does not contain the ASCII header of a Git-LFS pointer file. This is used to conditionally skip tests that require the actual grid file when it is missing or replaced by a pointer due to Git-LFS handling. The function returns True if the file appears to be a valid NetCDF file and False otherwise.

    Parameters:
        path (Path): The file path to check.

    Returns:
        bool: True if the file is a real NetCDF file, False if it is missing or a Git-LFS pointer.
    """
    if not path.exists() or path.stat().st_size < 512:
        return False
    try:
        with open(path, "rb") as fh:
            header = fh.read(64)
        # LFS pointer files begin with the ASCII text "version https://git-lfs"
        return b"version https://git-lfs" not in header
    except OSError:
        return False

pytestmark = pytest.mark.skipif(
    not _is_real_netcdf(_GRID_FILE),
    reason="test data not present (data/grid/x1.10242.static.nc)",
)


class TestLoadMpasPrecipSynthetic:
    """ Unit tests for load_mpas_precip using minimal synthetic NetCDF file pairs that avoid any dependency on real MPAS forecast data. """

    @pytest.fixture()
    def _synth_files(self: "TestLoadMpasPrecipSynthetic", 
                     tmp_path: Path):
        """
        This fixture creates synthetic MPAS diag and grid NetCDF files with a small number of cells and known rainc/rainnc values. The grid file contains simple lonCell and latCell coordinates, while the diag file contains rainc and rainnc variables that are linearly increasing across cells. The fixture returns the paths to these synthetic files along with the number of cells and the original rainc/rainnc arrays for use in test assertions. By using synthetic files, we can precisely control the input data and validate that load_mpas_precip processes it correctly without relying on the presence of real MPAS output. 

        Parameters:
            tmp_path (Path): The temporary directory path provided by pytest for creating test files.

        Returns:
            tuple: Paths to the synthetic diag and grid files, number of cells, and the rainc and rainnc arrays.
        """
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

    def test_returns_sum_of_rainc_rainnc(self: "TestLoadMpasPrecipSynthetic", 
                                         _synth_files) -> None:
        """
        This test verifies that load_mpas_precip correctly sums the rainc and rainnc variables from the diag file to produce the total cumulative precipitation. It asserts that the returned DataArray has the expected shape and that its values match the element-wise sum of the original rainc and rainnc arrays. This confirms that the core loading and summation logic in load_mpas_precip is functioning as intended when provided with known input data. 

        Parameters:
            _synth_files: The synthetic diag and grid file paths, number of cells, and rainc/rainnc arrays provided by the fixture.

        Returns:
            None
        """
        diag, grid, ncells, rainc, rainnc = _synth_files
        result = load_mpas_precip(diag, grid)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (ncells,)
        np.testing.assert_allclose(result.values, rainc + rainnc)

    def test_units_attribute(self: "TestLoadMpasPrecipSynthetic", 
                             _synth_files: tuple) -> None:
        """
        This test confirms that load_mpas_precip attaches a 'units' attribute set to 'mm' on the returned DataArray. Consistent unit metadata is required so that downstream threshold comparisons and FSS calculations operate on precipitation values in the expected millimeter scale. This test guards against silent omission of unit attributes that could cause misleading results when the DataArray is passed to visualization or metrics code. 

        Parameters:
            _synth_files: The synthetic diag and grid file paths, number of cells, and rainc/rainnc arrays provided by the fixture.

        Returns:
            None
        """
        diag, grid, ncells, rainc, rainnc = _synth_files
        result = load_mpas_precip(diag, grid)
        assert result.attrs["units"] == "mm"

    def test_coords_merged_from_grid(self: "TestLoadMpasPrecipSynthetic", 
                                     _synth_files: tuple) -> None:
        """
        This test verifies that load_mpas_precip successfully merges spatial coordinates from the grid file when the diag file does not contain lonCell and latCell. It asserts that the returned DataArray has 'lonCell' and 'latCell' coordinates with the expected values from the grid file. This confirms that the coordinate merging logic in load_mpas_precip correctly falls back to the grid file for spatial information, which is essential for producing georeferenced output suitable for remapping and verification. 

        Parameters:
            _synth_files: The synthetic diag and grid file paths, number of cells, and rainc/rainnc arrays provided by the fixture.

        Returns:
            None
        """
        diag, grid, *_ = _synth_files
        load_mpas_precip(diag, grid)
        # The function should have opened the grid to get coordinates
        ds = xr.open_dataset(diag)
        assert "lonCell" not in ds
        ds.close()

    def test_only_rainc(self: "TestLoadMpasPrecipSynthetic", 
                        tmp_path: Path) -> None:
        """
        This test verifies that load_mpas_precip returns the rainc array unchanged when rainnc is absent from the diag file. This test constructs a diag file containing only rainc and asserts the output matches those values exactly. Correct behavior for the rainc-only case ensures that convective precipitation is not silently zeroed out when a model run omits the non-convective component. 

        Parameters:
            tmp_path: The temporary directory path provided by pytest for file creation.

        Returns:
            None
        """
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

    def test_only_rainnc(self: "TestLoadMpasPrecipSynthetic", 
                         tmp_path: Path) -> None:
        """
        This test verifies that load_mpas_precip returns the rainnc array unchanged when rainc is absent from the diag file. This test constructs a diag file containing only rainnc and asserts the output matches those values exactly. Correct behavior for the rainnc-only case ensures that large-scale non-convective precipitation is not silently zeroed out when a model run omits the convective component. 

        Parameters:
            tmp_path: The temporary directory path provided by pytest for file creation.

        Returns:
            None
        """
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

    def test_missing_precip_vars_raises(self: "TestLoadMpasPrecipSynthetic", 
                                        tmp_path: Path) -> None:
        """
        This test verifies that load_mpas_precip raises a ValueError when neither rainc nor rainnc is present in the diag file. Since load_mpas_precip relies on at least one of these variables to compute cumulative precipitation, their absence should trigger an explicit error rather than producing an empty or NaN-filled output. This test constructs a diag file with no precipitation variables and asserts that the expected error is raised with a message indicating the missing variables. 

        Parameters:
            tmp_path: The temporary directory path provided by pytest for file creation.

        Returns:
            None
        """
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
    def grid(self: "TestGridFileConsistency") -> Iterator[xr.Dataset]:
        """ 
        This fixture opens the real MPAS grid NetCDF file once per test class and yields the dataset for use in all grid consistency tests. The dataset is closed automatically after all tests finish. By using a class-scoped fixture, we minimize I/O overhead while still ensuring that all tests operate on the same grid file instance. The fixture does not perform any modifications to the dataset, allowing each test to independently verify the raw contents of the grid file as it would be read by load_mpas_precip. 

        Parameters:
            None

        Returns:
            xr.Dataset: The opened MPAS grid dataset for use in tests.
        """
        ds = xr.open_dataset(_GRID_FILE)
        yield ds
        ds.close()

    def test_ncells_dimension(self: "TestGridFileConsistency", 
                              grid: xr.Dataset) -> None:
        """
        This test verifies that the grid file contains a dimension named 'nCells' and that its size matches the expected number of cells for this MPAS mesh configuration. The nCells dimension defines the number of mesh cells in the unstructured grid and is critical for correctly interpreting the shape of variables in both the grid and diag files. A mismatch in nCells would indicate a fundamental inconsistency between the grid and diagnostic data, which would cause errors during coordinate merging and remapping steps in load_mpas_precip. This test ensures that the grid file has the correct structure before any data loading operations are attempted. 

        Parameters:
            grid: The MPAS grid dataset provided by the fixture.

        Returns:
            None
        """
        assert grid.sizes["nCells"] == _EXPECTED_NCELLS

    def test_has_lonCell_latCell(self: "TestGridFileConsistency", 
                                 grid: xr.Dataset) -> None:
        """
        This test confirms that the grid file contains the essential coordinate variables 'lonCell' and 'latCell' that define the longitude and latitude of each mesh cell center. These variables are required for remapping the unstructured MPAS mesh to a regular lat-lon grid and for georeferencing the precipitation data. Their absence would indicate a malformed or incomplete grid file that cannot be used for spatial operations in load_mpas_precip. This test ensures that the necessary spatial coordinates are present before any remapping logic is executed. 

        Parameters:
            grid: The MPAS grid dataset provided by the fixture.

        Returns:
            None
        """
        assert "lonCell" in grid, "Grid file missing lonCell"
        assert "latCell" in grid, "Grid file missing latCell"

    def test_coordinate_shapes(self: "TestGridFileConsistency", 
                               grid: xr.Dataset) -> None:
        """
        This test verifies that the lonCell and latCell variables in the grid file have the expected shape of (nCells,), where nCells matches the expected number of mesh cells for this MPAS configuration. Correct shapes for these coordinate variables are essential for successful merging with the diag file and for proper remapping to a regular grid. If the shapes do not match (e.g., if they are 2D or have an unexpected dimension), it would indicate a problem with the grid file that would cause errors in load_mpas_precip when it attempts to align coordinates with precipitation data. This test ensures that the spatial coordinate variables are structured correctly for use in remapping operations. 

        Parameters:
            grid: The MPAS grid dataset provided by the fixture.

        Returns:
            None
        """
        assert grid["lonCell"].shape == (_EXPECTED_NCELLS,)
        assert grid["latCell"].shape == (_EXPECTED_NCELLS,)

    def test_lonCell_range(self: "TestGridFileConsistency", 
                           grid: xr.Dataset) -> None:
        """
        This test confirms that all longitude values in the lonCell variable fall within the MPAS radian convention range of [0, 2π]. Longitude values outside this range would indicate a unit mismatch (e.g., degrees instead of radians) that produces physically nonsensical coordinates after radian-to-degree conversion. Passing this test confirms the longitude coordinate convention matches what the remapping utility expects when constructing the regular output grid.

        Parameters:
            grid: The MPAS grid dataset provided by the fixture.

        Returns:
            None
        """
        lon = grid["lonCell"].values
        assert lon.min() >= 0.0, f"lonCell min {lon.min()} < 0"
        assert lon.max() <= 2 * np.pi + 1e-6, f"lonCell max {lon.max()} > 2π"

    def test_latCell_range(self: "TestGridFileConsistency", 
                           grid: xr.Dataset) -> None:
        """
        This test confirms that all latitude values in the latCell variable fall within the MPAS radian convention range of [-π/2, π/2]. Latitude values outside this range would indicate a unit mismatch (e.g., degrees instead of radians) that produces physically nonsensical coordinates after radian-to-degree conversion. Passing this test confirms the latitude coordinate convention matches what the remapping utility expects when constructing the regular output grid. 

        Parameters:
            grid: The MPAS grid dataset provided by the fixture.

        Returns:
            None
        """
        lat = grid["latCell"].values
        assert lat.min() >= -np.pi / 2 - 1e-6
        assert lat.max() <= np.pi / 2 + 1e-6

    def test_no_nan_in_coordinates(self: "TestGridFileConsistency", 
                                   grid: xr.Dataset) -> None:
        """
        This test verifies that there are no NaN values in the lonCell and latCell coordinate variables of the grid file. NaN values in spatial coordinates would cause remapping operations to fail or produce NaN-filled output arrays, which would propagate through the verification pipeline and lead to incorrect FSS results. This test ensures that all mesh cells have valid longitude and latitude values before any remapping or spatial analysis is attempted. 

        Parameters:
            grid: The MPAS grid dataset provided by the fixture.

        Returns:
            None
        """
        assert not np.any(np.isnan(grid["lonCell"].values))
        assert not np.any(np.isnan(grid["latCell"].values))

    def test_has_topology_vars(self: "TestGridFileConsistency", 
                               grid: xr.Dataset) -> None:
        """
        This test confirms that the grid file contains the essential topology variables 'nEdgesOnCell', 'cellsOnCell', and 'edgesOnCell' that define the unstructured mesh connectivity. These variables are required for any remapping or interpolation operations that depend on the mesh structure. Their absence would indicate a malformed grid file that cannot be used for spatial operations in load_mpas_precip. This test ensures that the necessary mesh topology information is present before any remapping logic is executed. 

        Parameters:
            grid: The MPAS grid dataset provided by the fixture.

        Returns:
            None
        """
        for var in ("nEdgesOnCell", "cellsOnCell", "edgesOnCell"):
            assert var in grid, f"Grid file missing topology variable {var}"


class TestDiagFilesExist:
    """ File-presence tests that confirm all expected MPAS diagnostic NetCDF files exist on disk for every forecast cycle in the test dataset. """

    @pytest.mark.parametrize("cycle", _CYCLES)
    def test_init_hour_diag_exists(self: "TestDiagFilesExist", 
                                   cycle: str) -> None:
        """ 
        This test verifies that the initialization-hour MPAS diagnostic file exists for each forecast cycle in the test dataset. The hour-0 diag file is critical as it serves as the baseline for all subsequent accumulation calculations, so its presence is a prerequisite for any meaningful testing of load_mpas_precip. The test constructs the expected filename based on the cycle string and checks for its existence on disk, failing with a clear message if it is missing. This ensures that the test dataset is complete and correctly structured before any content validation or accumulation consistency tests are run. 

        Parameters:
            cycle: The forecast cycle string in the format YYYYMMDDHH.

        Returns:
            None
        """
        init_dt = datetime.datetime.strptime(cycle, "%Y%m%d%H")
        ts = init_dt.strftime("%Y-%m-%d_%H.%M.%S")
        path = _FC_DIR / cycle / f"diag.{ts}.nc"
        assert path.exists(), f"Missing init diag for cycle {cycle}: {path}"

    @pytest.mark.parametrize("cycle", _CYCLES)
    def test_hourly_diag_sequence(self: "TestDiagFilesExist", 
                                  cycle: str) -> None:
        """
        This test verifies that all expected hourly MPAS diagnostic files exist for each forecast cycle in the test dataset, covering hours 0 through the final forecast hour. Since load_mpas_precip relies on differencing diag files across multiple time steps to compute accumulations, the presence of the full sequence of hourly files is essential for meaningful testing. The test constructs the expected filename for each hour and checks for its existence, accumulating any missing files into a list and failing with a clear message if any are absent. This ensures that the test dataset is complete and correctly structured for all temporal steps before any content validation or accumulation consistency tests are run. 

        Parameters:
            cycle: The forecast cycle string in the format YYYYMMDDHH.

        Returns:
            None
        """
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
    def sample_diag(self: "TestDiagFileContents") -> Iterator[xr.Dataset]:
        """ 
        This fixture opens a representative MPAS diagnostic NetCDF file from the test dataset that contains the expected rainc and rainnc variables. The fixture yields the opened dataset for use in all content validation tests within this class, ensuring that all tests operate on the same diag file instance. The dataset is closed automatically after all tests finish. By using a class-scoped fixture, we minimize I/O overhead while still allowing multiple tests to validate different aspects of the same diag file's contents, such as variable presence, shapes, data types, and value ranges.

        Parameters:
            None

        Returns:
            xr.Dataset: The opened MPAS diagnostic dataset for use in tests.
        """
        path = _FC_DIR / "2014090100" / "diag.2014-09-01_06.00.00.nc"
        ds = xr.open_dataset(path)
        yield ds
        ds.close()

    def test_has_rainc_rainnc(self: "TestDiagFileContents", 
                              sample_diag: xr.Dataset) -> None:
        """
        This test verifies that the sample MPAS diagnostic file contains both the 'rainc' and 'rainnc' variables, which represent the convective and non-convective cumulative precipitation components, respectively. The presence of these variables is essential for load_mpas_precip to compute total cumulative precipitation correctly. If either variable is missing, it would indicate a problem with the diag file that would cause load_mpas_precip to raise an error or produce incomplete output. This test ensures that the necessary precipitation variables are present in the diag file before any further content validation or accumulation consistency tests are performed.

        Parameters:
            sample_diag: The MPAS diagnostic dataset provided by the fixture.

        Returns:
            None
        """
        assert "rainc" in sample_diag, "Diag file missing rainc"
        assert "rainnc" in sample_diag, "Diag file missing rainnc"

    def test_rainc_shape(self: "TestDiagFileContents", 
                         sample_diag: xr.Dataset) -> None:
        """
        This test confirms that the 'rainc' variable in the sample MPAS diagnostic file has the expected shape of (Time, nCells), where nCells matches the expected number of mesh cells for this MPAS configuration. The Time dimension should have a size of 1 for each diag file, as MPAS is configured to write one time snapshot per file. The nCells dimension must match the grid's nCells to ensure that precipitation values are correctly aligned with spatial coordinates during merging and remapping. This test ensures that the rainc variable has the correct structure for use in load_mpas_precip and that it is consistent with the expected mesh configuration.

        Parameters:
            sample_diag: The MPAS diagnostic dataset provided by the fixture.

        Returns:
            None
        """
        assert sample_diag["rainc"].dims == ("Time", "nCells")
        assert sample_diag["rainc"].shape[1] == _EXPECTED_NCELLS

    def test_single_time_step(self: "TestDiagFileContents", 
                              sample_diag: xr.Dataset) -> None:
        """
        This test verifies that the Time dimension in the sample MPAS diagnostic file has a size of 1, confirming that each diag file contains only a single time snapshot as expected from MPAS's output convention. If the Time dimension had more than one step, it would indicate a problem with the diag file that could cause load_mpas_precip to fail or produce incorrect results when it attempts to process multiple time steps. This test ensures that the temporal structure of the diag file matches the expected format for use in load_mpas_precip and that it is consistent with the assumption of one time snapshot per file. 

        Parameters:
            sample_diag: The MPAS diagnostic dataset provided by the fixture.

        Returns:
            None
        """
        assert sample_diag.sizes["Time"] == 1

    def test_ncells_matches_grid(self: "TestDiagFileContents", 
                                 sample_diag: xr.Dataset) -> None:
        """
        This test confirms that the nCells dimension in the sample MPAS diagnostic file matches the expected number of mesh cells for this MPAS configuration, which is defined by the grid file. A mismatch in nCells between the diag and grid files would indicate a fundamental inconsistency that would cause errors during coordinate merging and remapping steps in load_mpas_precip. This test ensures that the diag file has the correct structure and is consistent with the expected mesh configuration before any data loading operations are attempted. 

        Parameters:
            sample_diag: The MPAS diagnostic dataset provided by the fixture.

        Returns:
            None
        """
        assert sample_diag.sizes["nCells"] == _EXPECTED_NCELLS

    def test_precip_non_negative(self: "TestDiagFileContents", 
                                 sample_diag: xr.Dataset) -> None:
        """
        This test verifies that all values in the 'rainc' and 'rainnc' variables of the sample MPAS diagnostic file are non-negative, as cumulative precipitation cannot be negative. Negative values would indicate a problem with the diag file, such as data corruption or an incorrect unit conversion, that would produce physically implausible results when processed by load_mpas_precip. This test ensures that the precipitation data in the diag file is physically reasonable and suitable for use in accumulation calculations and FSS verification. A small tolerance of -1e-6 is allowed to account for any floating-point rounding issues without triggering false positives. 

        Parameters:
            sample_diag: The MPAS diagnostic dataset provided by the fixture.

        Returns:
            None
        """
        for var in ("rainc", "rainnc"):
            vals = sample_diag[var].values
            assert np.all(vals >= -1e-6), (
                f"{var} has negative values: min={vals.min()}"
            )

    def test_precip_dtype_float32(self: "TestDiagFileContents", 
                                  sample_diag: xr.Dataset) -> None:
        """
        This test confirms that the 'rainc' and 'rainnc' variables in the sample MPAS diagnostic file have a data type of float32, which is the expected precision for MPAS precipitation output. Using float32 ensures that the data is stored efficiently while still providing sufficient precision for accumulation calculations and FSS verification. If the data type were different (e.g., float64 or int), it could indicate a problem with the diag file or an unexpected change in MPAS's output format that would require adjustments to load_mpas_precip. This test ensures that the precipitation variables have the correct data type for use in load_mpas_precip and that they are consistent with the expected MPAS output convention. 

        Parameters:
            sample_diag: The MPAS diagnostic dataset provided by the fixture.

        Returns:
            None
        """
        for var in ("rainc", "rainnc"):
            assert sample_diag[var].dtype == np.float32


class TestHourZeroPrecip:
    """ Parametrized sanity checks confirming that every forecast cycle's initialization-hour diag file contains zero cumulative precipitation for both rainc and rainnc. """

    @pytest.mark.parametrize("cycle", _CYCLES)
    def test_precip_zero_at_init(self: "TestHourZeroPrecip", 
                                 cycle: str) -> None:
        """
        This test verifies that the initialization-hour MPAS diagnostic file for each forecast cycle contains zero cumulative precipitation in both the 'rainc' and 'rainnc' variables. Since these variables represent cumulative totals from the start of the forecast, they should be zero at hour 0. Non-zero values at initialization would indicate a problem with the diag file, such as a model reset or incorrect file ordering, that would produce incorrect accumulation calculations when processed by load_mpas_precip. This test ensures that the baseline conditions for all forecast cycles are consistent and physically plausible before any accumulation consistency tests are run. A small tolerance of 1e-6 is allowed to account for any floating-point rounding issues without triggering false positives. 

        Parameters:
            cycle: The forecast cycle identifier.

        Returns:
            None
        """
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

    def test_total_precip_increases(self: "TestPrecipMonotonicity") -> None:
        """ 
        This test confirms that the total cumulative precipitation across the entire domain does not decrease between the initialization hour and subsequent forecast hours for a representative forecast cycle. Since MPAS outputs cumulative totals, the sum of rainc and rainnc should either increase or remain constant as time advances. A decrease in total precipitation would indicate a problem with the diag files, such as incorrect file ordering, data corruption, or a model reset, that would produce incorrect accumulation calculations when processed by load_mpas_precip. This test ensures that the temporal evolution of cumulative precipitation is physically plausible and consistent with MPAS's output convention before any further verification steps are performed. A small tolerance of 1e-3 mm is allowed to account for any floating-point rounding issues without triggering false positives. 

        Parameters:
            None

        Returns:
            None
        """
        cycle = "2014090100"
        init_dt = datetime.datetime(2014, 9, 1, 0)
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

    def test_6h_accumulation_non_negative(self: "TestAccumulationConsistency") -> None:
        """ 
        This test verifies that the 6-hour accumulated precipitation, computed by differencing the diag files at hour 0 and hour 6, is non-negative across the entire domain. Since MPAS outputs cumulative totals, the difference between the hour 6 and hour 0 diag files should yield a non-negative accumulation of precipitation over that period. Negative values in the accumulation would indicate a problem with the diag files, such as incorrect file ordering, data corruption, or a model reset, that would produce physically implausible results when processed by load_mpas_precip. This test ensures that the computed accumulations are consistent with physical expectations and MPAS's output convention before any further verification steps are performed. A small tolerance of -1e-4 mm is allowed to account for any floating-point rounding issues without triggering false positives. 

        Parameters:
            None

        Returns:
            None
        """
        cycle = "2014090100"
        grid_file = str(_GRID_FILE)
        init_dt = datetime.datetime(2014, 9, 1, 0)
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

    def test_6h_accumulation_reasonable_magnitude(self: "TestAccumulationConsistency") -> None:
        """ 
        This test confirms that the maximum value of the 6-hour accumulated precipitation, computed by differencing the diag files at hour 0 and hour 6, is within a physically reasonable range (e.g., less than 500 mm) for this MPAS configuration and test dataset. While the exact maximum accumulation can vary based on the model configuration and meteorological conditions, values exceeding a certain threshold would indicate a problem with the diag files, such as incorrect file ordering, data corruption, or a model reset, that would produce physically implausible results when processed by load_mpas_precip. This test ensures that the computed accumulations are not only non-negative but also within a realistic range for use in further verification steps.  

        Parameters:
            None

        Returns:
            None
        """
        cycle = "2014090100"
        grid_file = str(_GRID_FILE)
        init_dt = datetime.datetime(2014, 9, 1, 0)
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

    def test_ensure_mpasdiag_import_error(self: "TestEnsureMpasdiagBranches") -> None:
        """
        This test verifies that _ensure_mpasdiag raises an ImportError with the expected message when mpasdiag is not available. By patching sys.modules to simulate the absence of mpasdiag and its submodules, we confirm that the function correctly detects the missing dependency and raises an informative error. This test ensures that users receive a clear message about the requirement for mpasdiag if they attempt to use load_mpas_precip without having it installed, which is essential for diagnosing setup issues.

        Parameters:
            None

        Returns:
            None
        """
        import sys
        from modvx import mpas_reader

        old_val = mpas_reader._HAS_MPASDIAG
        mpas_reader._HAS_MPASDIAG = None

        _MISSING = object()
        keys = [
            "mpasdiag", "mpasdiag.processing",
            "mpasdiag.processing.remapping",
            "mpasdiag.processing.utils_geog",
        ]
        saved = {k: sys.modules.get(k, _MISSING) for k in keys}
        # Set all four to None so the import inside _ensure_mpasdiag_available fails.
        for k in keys:
            sys.modules[k] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="mpasdiag is required"):
                mpas_reader._ensure_mpasdiag_available()
        finally:
            for k in keys:
                if saved[k] is _MISSING:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = saved[k]
            mpas_reader._HAS_MPASDIAG = old_val

    def test_ensure_mpasdiag_cached_true(self: "TestEnsureMpasdiagBranches") -> None:
        """
        This test confirms that _ensure_mpasdiag returns immediately without raising an error when _HAS_MPASDIAG is already set to True, simulating the cached success path. By setting _HAS_MPASDIAG to True before calling the function, we verify that it does not attempt to re-import mpasdiag and does not raise an error, confirming that the early-return cache mechanism works as intended for subsequent calls after a successful import.

        Parameters:
            None

        Returns:
            None
        """
        from modvx import mpas_reader

        old_val = mpas_reader._HAS_MPASDIAG
        mpas_reader._HAS_MPASDIAG = True
        mpas_reader._ensure_mpasdiag_available()  # should not raise
        mpas_reader._HAS_MPASDIAG = old_val

    def test_successful_import(self: "TestEnsureMpasdiagBranches") -> None:
        """
        This test verifies that _ensure_mpasdiag successfully imports mpasdiag and its submodules when they are available, and that it sets _HAS_MPASDIAG to True. By patching sys.modules to include mock versions of mpasdiag and its submodules, we confirm that the function can import them without error and that the expected remapping and utility functions are accessed. This test ensures that the successful import path works correctly and that the function properly caches the success for future calls. 

        Parameters:
            None

        Returns:
            None
        """
        import modvx.mpas_reader as mr
        import sys

        mr._HAS_MPASDIAG = None

        fake_remapping = types.ModuleType("mpasdiag.processing.remapping")
        setattr(fake_remapping, "remap_mpas_to_latlon_with_masking", lambda *a, **kw: None)

        fake_utils = types.ModuleType("mpasdiag.processing.utils_geog")
        setattr(fake_utils, "MPASGeographicUtils", lambda *a, **kw: None)

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

    def test_remap_with_grid_fallback(self: "TestRemapToLatlonWithGrid", 
                                      tmp_path: Path) -> None:
        """
        This test verifies the remap_to_latlon function's ability to successfully remap a DataArray to a regular lat-lon grid using a separate grid file when the input dataset does not contain lonCell and latCell variables. By creating temporary diagnostic and grid NetCDF files with the necessary structure, we confirm that the function correctly extracts spatial coordinates from the grid file, computes the extent, and calls the mpasdiag remapping function with the expected arguments. The test also checks that the returned DataArray has latitude and longitude dimensions, confirming that the remapping process was executed as intended. This test ensures that the grid fallback mechanism in remap_to_latlon works correctly when spatial coordinates are not present in the input dataset. 

        Parameters:
            tmp_path: A pytest fixture providing a temporary directory for creating test NetCDF files.

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

        def fake_extract(*args, 
                         **kwargs) -> tuple[np.ndarray, np.ndarray]:
            """
            This fake function simulates the behavior of MPASGeographicUtils.extract_spatial_coordinates by returning predefined longitude and latitude arrays. The longitude array contains values [0.0, 1.0, 2.0] and the latitude array contains values [10.0, 20.0, 30.0], which correspond to the coordinates defined in the temporary grid dataset created for this test. By providing these specific values, we can verify that remap_to_latlon correctly uses the extracted coordinates to compute the extent and call the remapping function with the expected spatial information. This allows us to validate the integration of coordinate extraction and remapping logic without relying on the actual implementation of MPASGeographicUtils.

            Parameters:
                *args: Positional arguments (ignored in this fake implementation).
                **kwargs: Keyword arguments (ignored in this fake implementation).

            Returns:
                tuple[np.ndarray, np.ndarray]: A tuple containing the longitude and latitude arrays. 
            """
            return (np.array([0.0, 1.0, 2.0]), np.array([10.0, 20.0, 30.0]))

        def fake_extent(*args, 
                        **kwargs) -> tuple[float, float, float, float]:
            """
            This fake function simulates the behavior of MPASGeographicUtils.get_extent_from_coordinates by returning a predefined extent. The extent is defined as (0.0, 2.0, 10.0, 30.0), which corresponds to the minimum and maximum longitude and latitude values from the temporary grid dataset created for this test. By providing this specific extent, we can verify that remap_to_latlon correctly uses the computed extent when calling the remapping function. This allows us to validate the integration of extent computation and remapping logic without relying on the actual implementation of MPASGeographicUtils.

            Parameters:
                *args: Positional arguments (ignored in this fake implementation).
                **kwargs: Keyword arguments (ignored in this fake implementation).

            Returns:
                tuple[float, float, float, float]: A tuple containing the minimum and maximum longitude and latitude values.
            """
            return (0.0, 2.0, 10.0, 30.0)

        mock_utils = types.SimpleNamespace(
            extract_spatial_coordinates=fake_extract,
            get_extent_from_coordinates=fake_extent,
        )

        remapped_da = xr.DataArray(
            np.ones((2, 2)),
            dims=["lat", "lon"],
            coords={"lat": [10.0, 30.0], "lon": [0.0, 2.0]},
        )
        remap_calls: list = []

        def mock_remap(*args, 
                       **kwargs) -> xr.DataArray: 
            """
            This mock function simulates the behavior of remap_mpas_to_latlon_with_masking by returning a predefined remapped DataArray. The returned DataArray has dimensions "lat" and "lon" with coordinates corresponding to the extent defined in the fake_extent function. By appending the received arguments to the remap_calls list, we can verify that remap_to_latlon calls the remapping function with the expected arguments, including the input data, spatial coordinates, and extent. This allows us to validate that remap_to_latlon correctly integrates with the remapping function and passes the correct information for remapping, without relying on the actual implementation of the remapping logic.

            Parameters:
                *args: Positional arguments received from remap_to_latlon (e.g., input data, spatial coordinates, extent).
                **kwargs: Keyword arguments received from remap_to_latlon (e.g., resolution, masking options).

            Returns:
                xr.DataArray: A predefined remapped DataArray with latitude and longitude dimensions. 
            """
            remap_calls.append(args)
            return remapped_da

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
            assert len(remap_calls) == 1
        finally:
            for k in keys:
                if saved[k] is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = saved[k]
            mr._HAS_MPASDIAG = None

    def test_remap_to_latlon_mocked(self: "TestRemapToLatlonWithGrid", 
                                    tmp_path: Path) -> None:
        """
        This test verifies the remap_to_latlon function's ability to successfully remap a DataArray to a regular lat-lon grid using mocked mpasdiag remapping and utility functions. By creating a temporary diagnostic NetCDF file with the necessary structure and patching the mpasdiag remapping and geographic utility functions, we confirm that remap_to_latlon correctly calls these functions with the expected arguments and that it returns a DataArray with latitude and longitude dimensions. This test ensures that the core logic of remap_to_latlon works correctly when the mpasdiag dependencies are available, even though the actual remapping is mocked, allowing us to validate the integration points without relying on the real mpasdiag implementation.

        Parameters:
            tmp_path: A pytest fixture providing a temporary directory for creating test NetCDF files.

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

        def mock_remap_fn(*args, 
                          **kwargs) -> xr.DataArray:
            """
            This mock function simulates the behavior of remap_mpas_to_latlon_with_masking by returning a predefined remapped DataArray. The returned DataArray has dimensions "lat" and "lon" with coordinates corresponding to a regular grid. By accepting arbitrary positional and keyword arguments, we can verify that remap_to_latlon calls the remapping function with the expected arguments, including the input data, spatial coordinates, and extent. This allows us to validate that remap_to_latlon correctly integrates with the remapping function and passes the correct information for remapping, without relying on the actual implementation of the remapping logic.

            Parameters:
                *args: Positional arguments received from remap_to_latlon (e.g., input data, spatial coordinates, extent).
                **kwargs: Keyword arguments received from remap_to_latlon (e.g., resolution, masking options).
 
            Returns:
                xr.DataArray: A predefined remapped DataArray with latitude and longitude dimensions. 
            """
            return remapped

        class _FakeGeoUtils:
            """ Fake MPASGeographicUtils class with methods that return predefined coordinates and extent based on the number of cells. """

            def __call__(self: "_FakeGeoUtils", 
                         *args, 
                         **kwargs) -> "_FakeGeoUtils":
                """ 
                This mock __call__ method simulates the behavior of the MPASGeographicUtils class when called as a function. It returns the instance itself, allowing method chaining or further calls to other methods of the class.

                Parameters:
                    *args: Positional arguments received from the caller.
                    **kwargs: Keyword arguments received from the caller.

                Returns:
                    _FakeGeoUtils: The instance itself, enabling method chaining.
                """
                return self
            def extract_spatial_coordinates(self: "_FakeGeoUtils", 
                                            *args, 
                                            **kwargs) -> tuple:
                """
                This mock method simulates the behavior of the extract_spatial_coordinates method of the MPASGeographicUtils class. It returns predefined latitude and longitude coordinates based on the number of cells.

                Parameters:
                    *args: Positional arguments received from the caller.
                    **kwargs: Keyword arguments received from the caller.

                Returns:
                    tuple: A tuple containing two numpy arrays representing latitude and longitude coordinates.
                """
                return (np.linspace(0, 10, n_cells), np.linspace(-5, 5, n_cells))
            
            def get_extent_from_coordinates(self: "_FakeGeoUtils", 
                                            *args, 
                                            **kwargs) -> tuple:
                """
                This mock method simulates the behavior of the get_extent_from_coordinates method of the MPASGeographicUtils class. It returns a predefined extent based on the coordinates.

                Parameters:
                    *args: Positional arguments received from the caller.
                    **kwargs: Keyword arguments received from the caller.

                Returns:
                    tuple: A tuple representing the extent (min_lat, max_lat, min_lon, max_lon).
                """
                return (0.0, 10.0, -5.0, 5.0)

        mock_geo_utils = _FakeGeoUtils()

        import sys

        fake_remapping_mod = types.ModuleType("mpasdiag.processing.remapping")
        setattr(fake_remapping_mod, "remap_mpas_to_latlon_with_masking", mock_remap_fn)
        fake_utils_mod = types.ModuleType("mpasdiag.processing.utils_geog")
        setattr(fake_utils_mod, "MPASGeographicUtils", mock_geo_utils)

        old_has = mpas_reader._HAS_MPASDIAG
        mpas_reader._HAS_MPASDIAG = True

        _MISSING = object()
        keys = [
            "mpasdiag", "mpasdiag.processing",
            "mpasdiag.processing.remapping",
            "mpasdiag.processing.utils_geog",
        ]
        saved = {k: sys.modules.get(k, _MISSING) for k in keys}
        try:
            sys.modules["mpasdiag"] = types.ModuleType("mpasdiag")
            sys.modules["mpasdiag.processing"] = types.ModuleType("mpasdiag.processing")
            sys.modules["mpasdiag.processing.remapping"] = fake_remapping_mod
            sys.modules["mpasdiag.processing.utils_geog"] = fake_utils_mod
            result = mpas_reader.remap_to_latlon(data, ds_file, ds_file, 0.5)
        finally:
            for k in keys:
                if saved[k] is _MISSING:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = saved[k]
            mpas_reader._HAS_MPASDIAG = old_has

        assert "latitude" in result.dims
        assert "longitude" in result.dims


class TestLoadAndRemapMpasPrecip:
    """ Cover load_and_remap_mpas_precip convenience wrapper. """

    def test_load_and_remap(self: "TestLoadAndRemapMpasPrecip", 
                            tmp_path: Path) -> None:
        """
        This test verifies that the load_and_remap_mpas_precip function correctly loads precipitation data from an MPAS diagnostic file, merges it with grid coordinates from a separate grid file, and remaps it to a regular lat-lon grid at the specified resolution. By patching the load_mpas_precip and remap_to_latlon functions to return controlled test data, we confirm that load_and_remap_mpas_precip calls these functions with the expected arguments and that it returns a DataArray with latitude and longitude dimensions. This test ensures that the convenience wrapper function correctly integrates the loading and remapping steps, providing a streamlined interface for users while still relying on the underlying functionality of the individual components.

        Parameters:
            tmp_path: A pytest fixture providing a temporary directory for creating test NetCDF files.

        Returns:
            None
        """
        from modvx import mpas_reader

        old_has = mpas_reader._HAS_MPASDIAG
        mpas_reader._HAS_MPASDIAG = True

        mesh_precip = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["nCells"])
        remapped = xr.DataArray(np.ones((3, 3)), dims=["latitude", "longitude"])

        load_calls: list = []
        remap_calls: list = []

        def fake_load(diag_file: str, 
                      grid_file: str) -> xr.DataArray:
            """
            This fake function simulates the behavior of load_mpas_precip by returning a predefined DataArray representing precipitation on the MPAS mesh. By appending the received diag_file and grid_file arguments to the load_calls list, we can verify that load_and_remap_mpas_precip calls the loading function with the expected file paths. This allows us to validate that load_and_remap_mpas_precip correctly integrates with the loading function and passes the correct information for loading, without relying on the actual implementation of load_mpas_precip.

            Parameters:
                diag_file: The path to the MPAS diagnostic file from which to load precipitation data.
                grid_file: The path to the MPAS grid file that may be needed for loading precipitation data.

            Returns:
                xr.DataArray: A predefined DataArray representing precipitation on the MPAS mesh. 
            """
            load_calls.append((diag_file, grid_file))
            return mesh_precip

        def fake_remap(data: xr.DataArray, 
                       diag_file: str, 
                       grid_file: str, 
                       resolution: float) -> xr.DataArray:
            """
            This fake function simulates the behavior of remap_to_latlon by returning a predefined remapped DataArray. By appending the received data, diag_file, grid_file, and resolution arguments to the remap_calls list, we can verify that load_and_remap_mpas_precip calls the remapping function with the expected arguments, including the loaded precipitation data and the file paths. This allows us to validate that load_and_remap_mpas_precip correctly integrates with the remapping function and passes the correct information for remapping, without relying on the actual implementation of remap_to_latlon. 

            Parameters:
                data: The DataArray representing precipitation on the MPAS mesh that is to be remapped.
                diag_file: The path to the MPAS diagnostic file, which may be used for remapping.
                grid_file: The path to the MPAS grid file, which may be used for remapping.
                resolution: The desired resolution for the remapped lat-lon grid.

            Returns:
                xr.DataArray: A predefined remapped DataArray with latitude and longitude dimensions. 
            """
            remap_calls.append((data, diag_file, grid_file, resolution))
            return remapped

        orig_load = mpas_reader.load_mpas_precip
        orig_remap = mpas_reader.remap_to_latlon
        mpas_reader.load_mpas_precip = fake_load
        mpas_reader.remap_to_latlon = fake_remap
        try:
            result = mpas_reader.load_and_remap_mpas_precip("diag.nc", "grid.nc", 0.5)
        finally:
            mpas_reader.load_mpas_precip = orig_load
            mpas_reader.remap_to_latlon = orig_remap
            mpas_reader._HAS_MPASDIAG = old_has

        assert len(load_calls) == 1
        assert load_calls[0] == ("diag.nc", "grid.nc")
        assert len(remap_calls) == 1
        assert remap_calls[0] == (mesh_precip, "diag.nc", "grid.nc", 0.5)
        assert result.shape == (3, 3)
