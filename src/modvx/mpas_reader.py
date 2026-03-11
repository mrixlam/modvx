#!/usr/bin/env python3

"""
MPAS unstructured-mesh forecast reader for modvx.

This module defines the MPASReader class, which is responsible for loading forecast data from MPAS diagnostic NetCDF files. The reader handles the specific structure and conventions of MPAS output, including parsing valid times from filenames, normalizing coordinate systems, and extracting relevant variables for verification. By encapsulating MPAS-specific logic in this class, we can maintain a clean separation between data access and the core verification algorithms, allowing for easier maintenance and potential extension to other model formats in the future.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional

import xarray as xr

logger = logging.getLogger(__name__)

# Lazy import flag — avoid hard dependency when not using MPAS mesh data
_HAS_MPASDIAG: Optional[bool] = None


def _variables_to_drop(filepath: str, keep: tuple[str, ...]) -> list[str]:
    """
    Open a NetCDF file and return the list of variable names present in the file
    that are not included in the provided *keep* tuple. Useful for building the
    `drop_variables` argument to xarray loading routines so only necessary
    variables are read into memory.

    Parameters:
        filepath (str): Path to the NetCDF file to probe.
        keep (tuple of str): Tuple of variable names to retain.

    Returns:
        list[str]: Variable names present in the file that are not in *keep*.
    """
    with xr.open_dataset(filepath) as probe:
        return [str(var) for var in probe.data_vars if var not in keep]


def _ensure_mpasdiag_available() -> None:
    """
    Lazily import required mpasdiag modules on first use and raise a descriptive
    error if unavailable. The import cache is stored in the module-level
    ``_HAS_MPASDIAG`` flag so subsequent calls are inexpensive. Raises an
    informative ImportError with installation instructions when mpasdiag is
    not present.

    Returns:
        None

    Raises:
        ImportError: If the ``mpasdiag`` package cannot be imported.
    """
    global _HAS_MPASDIAG
    if _HAS_MPASDIAG is True:
        return
    try:
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking  # type: ignore[import-untyped]  # noqa: F401
        from mpasdiag.processing.utils_geog import MPASGeographicUtils  # type: ignore[import-untyped]  # noqa: F401
        _HAS_MPASDIAG = True
    except ImportError as exc:
        _HAS_MPASDIAG = False
        raise ImportError(
            "mpasdiag is required for MPAS mesh forecast reading. "
            "Install it with: pip install -e /path/to/MPASdiag"
        ) from exc


def load_mpas_precip(
    diag_file: str,
    grid_file: str,
) -> xr.DataArray:
    """
    Load total (cumulative) precipitation from a single MPAS diagnostic file on the native mesh. Both convective (``rainc``) and non-convective (``rainnc``) precipitation components are read and summed, with fallback to whichever component is available when only one is present. Grid coordinates (``lonCell``, ``latCell``) are merged from the grid file if they are absent from the diagnostic file. The returned array remains on the unstructured MPAS mesh; call remap_to_latlon for conversion to a regular grid.

    Parameters:
        diag_file (str): Path to an MPAS ``diag.*`` NetCDF file containing ``rainc``
            and/or ``rainnc``.
        grid_file (str): Path to the MPAS grid file that provides ``lonCell``/``latCell``
            coordinates.

    Returns:
        xr.DataArray: Total cumulative precipitation on the MPAS mesh (``nCells`` dimension)
            in millimetres, with ``lonCell``/``latCell`` coordinates attached.
    """
    # Only load precipitation variables to minimise memory for high-res meshes
    with xr.open_dataset(diag_file) as probe:
        available_vars = set(probe.data_vars)
    precip_components = [v for v in ("rainc", "rainnc") if v in available_vars]
    if not precip_components:
        raise ValueError(
            f"No 'rainc' or 'rainnc' variable in {diag_file}"
        )
    dataset = xr.load_dataset(diag_file, drop_variables=[
        v for v in available_vars if v not in precip_components
    ])

    # Merge grid coordinates if not present in diag file
    if "lonCell" not in dataset and "latCell" not in dataset:
        grid_dataset = xr.load_dataset(
            grid_file, drop_variables=_variables_to_drop(grid_file, ("lonCell", "latCell")),
        )
        dataset["lonCell"] = grid_dataset["lonCell"]
        dataset["latCell"] = grid_dataset["latCell"]
        grid_dataset.close()

    # Compute total precip from rainc + rainnc
    if "rainc" in dataset and "rainnc" in dataset:
        total_precip = dataset["rainc"].isel(Time=0) + dataset["rainnc"].isel(Time=0)
    elif "rainc" in dataset:
        total_precip = dataset["rainc"].isel(Time=0)
    else:
        total_precip = dataset["rainnc"].isel(Time=0)

    total_precip.attrs["units"] = "mm"
    total_precip.attrs["_mpas_dataset_path"] = diag_file
    # Release the full dataset immediately
    dataset.close()
    del dataset
    return total_precip


def remap_to_latlon(
    data: xr.DataArray,
    dataset_or_file: str,
    grid_file: str,
    resolution: float = 0.1,
) -> xr.DataArray:
    """
    Remap a field from the unstructured MPAS mesh to a regular latitude-longitude grid. Geographic bounds are automatically determined from the mesh coordinates via MPASGeographicUtils, so no domain information needs to be supplied explicitly. Remapping is performed using ``remap_mpas_to_latlon_with_masking`` from mpasdiag with nearest-neighbour interpolation. After remapping, dimension names are standardised to ``latitude``/``longitude`` for pipeline compatibility.

    Parameters:
        data (xr.DataArray): Precipitation or other field on the unstructured MPAS mesh
            (``nCells`` dimension).
        dataset_or_file (str): Path to the MPAS dataset file providing coordinate variables
            for bounds extraction.
        grid_file (str): Path to the MPAS grid file containing ``lonCell``/``latCell`` if
            not in the dataset file.
        resolution (float): Target grid spacing in degrees (default: 0.1°).

    Returns:
        xr.DataArray: Remapped field on a regular lat-lon grid with ``latitude`` and
            ``longitude`` coordinates.
    """
    _ensure_mpasdiag_available()
    from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking  # type: ignore[import-untyped]
    from mpasdiag.processing.utils_geog import MPASGeographicUtils  # type: ignore[import-untyped]

    # Only load coordinate variables needed for geographic bounds extraction
    coord_vars_to_keep = ("lonCell", "latCell")
    dataset = xr.load_dataset(
        dataset_or_file, drop_variables=_variables_to_drop(dataset_or_file, coord_vars_to_keep),
    )
    if "lonCell" not in dataset:
        grid_dataset = xr.load_dataset(
            grid_file, drop_variables=_variables_to_drop(grid_file, coord_vars_to_keep),
        )
        dataset["lonCell"] = grid_dataset["lonCell"]
        dataset["latCell"] = grid_dataset["latCell"]
        grid_dataset.close()

    # Extract geographic bounds from the grid
    lon, lat = MPASGeographicUtils.extract_spatial_coordinates(dataset, normalize=False)
    lon_min, lon_max, lat_min, lat_max = (
        MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=0.0)
    )

    logger.debug(
        "MPAS remapping: extent [%.2f, %.2f] x [%.2f, %.2f], res=%.3f°",
        lon_min, lon_max, lat_min, lat_max, resolution,
    )

    remapped_da = remap_mpas_to_latlon_with_masking(
        data=data,
        dataset=dataset,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        resolution=resolution,
        method="nearest",
        apply_mask=True,
        lon_convention="auto",
    )

    dataset.close()

    # Standardise coordinate names to latitude/longitude
    if "lat" in remapped_da.dims and "latitude" not in remapped_da.dims:
        remapped_da = remapped_da.rename({"lat": "latitude"})
    if "lon" in remapped_da.dims and "longitude" not in remapped_da.dims:
        remapped_da = remapped_da.rename({"lon": "longitude"})

    return remapped_da


def load_and_remap_mpas_precip(
    diag_file: str,
    grid_file: str,
    resolution: float = 0.1,
) -> xr.DataArray:
    """
    Load total precipitation from an MPAS diag file and remap it to a regular lat-lon grid. This is the primary public entry point for MPAS forecast data, combining load_mpas_precip and remap_to_latlon into a single convenience call. It is used by FileManager.accumulate_forecasts when processing individual MPAS time steps. The mpasdiag library is required and will be lazily imported on the first call via ``_ensure_mpasdiag_available``.

    Parameters:
        diag_file (str): Path to an MPAS ``diag.*`` NetCDF file containing precipitation
            variables.
        grid_file (str): Path to the MPAS grid file for coordinate and bounds information.
        resolution (float): Target grid spacing for the output lat-lon grid in degrees
            (default: 0.1°).

    Returns:
        xr.DataArray: Total precipitation on a regular lat-lon grid in millimetres.
    """
    _ensure_mpasdiag_available()
    precip_mesh = load_mpas_precip(diag_file, grid_file)
    return remap_to_latlon(precip_mesh, diag_file, grid_file, resolution)
