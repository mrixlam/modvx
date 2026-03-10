"""
MPAS unstructured-mesh forecast reader for modvx.

Uses ``mpasdiag`` to load native MPAS diagnostic files and remap
precipitation fields (``rainc + rainnc``) to a regular lat-lon grid
suitable for verification.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Lazy import flag — avoid hard dependency when not using MPAS mesh data
_HAS_MPASDIAG: Optional[bool] = None


def _ensure_mpasdiag() -> None:
    """
    Lazily import required mpasdiag modules on first use and raise a descriptive error if unavailable.
    The import is cached via a module-level flag so that repeated calls only incur a dictionary
    lookup rather than re-running the import machinery. When mpasdiag is not installed, a clear
    ImportError with installation instructions is raised to help users resolve the dependency.
    This function must be called at the top of any function that uses mpasdiag functionality.

    Returns:
        None

    Raises:
        ImportError: If the ``mpasdiag`` package cannot be imported.
    """
    global _HAS_MPASDIAG
    if _HAS_MPASDIAG is True:
        return
    try:
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking  # noqa: F401
        from mpasdiag.processing.utils_geog import MPASGeographicUtils  # noqa: F401
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
    Load total (cumulative) precipitation from a single MPAS diagnostic file on the native mesh.
    Both convective (``rainc``) and non-convective (``rainnc``) precipitation components are
    read and summed, with fallback to whichever component is available when only one is present.
    Grid coordinates (``lonCell``, ``latCell``) are merged from the grid file if they are absent
    from the diagnostic file. The returned array remains on the unstructured MPAS mesh;
    call remap_to_latlon for conversion to a regular grid.

    Parameters:
        diag_file (str): Path to an MPAS ``diag.*`` NetCDF file containing ``rainc``
            and/or ``rainnc``.
        grid_file (str): Path to the MPAS grid file that provides ``lonCell``/``latCell``
            coordinates.

    Returns:
        xr.DataArray: Total cumulative precipitation on the MPAS mesh (``nCells`` dimension)
            in millimetres, with ``lonCell``/``latCell`` coordinates attached.
    """
    ds = xr.load_dataset(diag_file)

    # Merge grid coordinates if not present in diag file
    if "lonCell" not in ds and "latCell" not in ds:
        grid_ds = xr.load_dataset(grid_file)
        ds["lonCell"] = grid_ds["lonCell"]
        ds["latCell"] = grid_ds["latCell"]
        grid_ds.close()

    # Compute total precip from rainc + rainnc
    if "rainc" in ds and "rainnc" in ds:
        total = ds["rainc"].isel(Time=0) + ds["rainnc"].isel(Time=0)
    elif "rainc" in ds:
        total = ds["rainc"].isel(Time=0)
    elif "rainnc" in ds:
        total = ds["rainnc"].isel(Time=0)
    else:
        raise ValueError(
            f"No 'rainc' or 'rainnc' variable in {diag_file}"
        )

    total.attrs["units"] = "mm"
    # Keep the parent dataset reference for remapping
    total.attrs["_mpas_dataset_path"] = diag_file
    ds.close()
    return total


def remap_to_latlon(
    data: xr.DataArray,
    dataset_or_file: str,
    grid_file: str,
    resolution: float = 0.1,
) -> xr.DataArray:
    """
    Remap a field from the unstructured MPAS mesh to a regular latitude-longitude grid.
    Geographic bounds are automatically determined from the mesh coordinates via
    MPASGeographicUtils, so no domain information needs to be supplied explicitly.
    Remapping is performed using ``remap_mpas_to_latlon_with_masking`` from mpasdiag
    with nearest-neighbour interpolation. After remapping, dimension names are
    standardised to ``latitude``/``longitude`` for pipeline compatibility.

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
    _ensure_mpasdiag()
    from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking
    from mpasdiag.processing.utils_geog import MPASGeographicUtils

    # Open dataset to get coordinates
    ds = xr.load_dataset(dataset_or_file)
    if "lonCell" not in ds:
        grid_ds = xr.load_dataset(grid_file)
        ds["lonCell"] = grid_ds["lonCell"]
        ds["latCell"] = grid_ds["latCell"]
        grid_ds.close()

    # Extract geographic bounds from the grid
    lon, lat = MPASGeographicUtils.extract_spatial_coordinates(ds, normalize=False)
    lon_min, lon_max, lat_min, lat_max = (
        MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=0.0)
    )

    logger.debug(
        "MPAS remapping: extent [%.2f, %.2f] x [%.2f, %.2f], res=%.3f°",
        lon_min, lon_max, lat_min, lat_max, resolution,
    )

    remapped = remap_mpas_to_latlon_with_masking(
        data=data,
        dataset=ds,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        resolution=resolution,
        method="nearest",
        apply_mask=True,
        lon_convention="auto",
    )

    ds.close()

    # Standardise coordinate names to latitude/longitude
    if "lat" in remapped.dims and "latitude" not in remapped.dims:
        remapped = remapped.rename({"lat": "latitude"})
    if "lon" in remapped.dims and "longitude" not in remapped.dims:
        remapped = remapped.rename({"lon": "longitude"})

    return remapped


def load_and_remap_mpas_precip(
    diag_file: str,
    grid_file: str,
    resolution: float = 0.1,
) -> xr.DataArray:
    """
    Load total precipitation from an MPAS diag file and remap it to a regular lat-lon grid.
    This is the primary public entry point for MPAS forecast data, combining load_mpas_precip
    and remap_to_latlon into a single convenience call. It is used by
    FileManager.accumulate_forecasts when processing individual MPAS time steps. The
    mpasdiag library is required and will be lazily imported on the first call via
    _ensure_mpasdiag.

    Parameters:
        diag_file (str): Path to an MPAS ``diag.*`` NetCDF file containing precipitation
            variables.
        grid_file (str): Path to the MPAS grid file for coordinate and bounds information.
        resolution (float): Target grid spacing for the output lat-lon grid in degrees
            (default: 0.1°).

    Returns:
        xr.DataArray: Total precipitation on a regular lat-lon grid in millimetres.
    """
    _ensure_mpasdiag()
    precip_mesh = load_mpas_precip(diag_file, grid_file)
    return remap_to_latlon(precip_mesh, diag_file, grid_file, resolution)
