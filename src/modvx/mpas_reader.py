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
import xarray as xr
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level cache flag for mpasdiag availability to avoid repeated import attempts
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
    # Return the list of variable names in the file that are not in the keep tuple
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
    # Check the cached availability flag to avoid repeated import attempts
    global _HAS_MPASDIAG

    # If the availability has already been determined, return immediately
    if _HAS_MPASDIAG is True:
        return
    
    # If the availability has already been determined to be False, raise the same ImportError again
    try:
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking  # type: ignore[import-untyped]  
        from mpasdiag.processing.utils_geog import MPASGeographicUtils  # type: ignore[import-untyped] 
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
        diag_file (str): Path to an MPAS ``diag.*`` NetCDF file containing ``rainc`` and/or ``rainnc``.
        grid_file (str): Path to the MPAS grid file that provides ``lonCell``/``latCell`` coordinates.

    Returns:
        xr.DataArray: Total cumulative precipitation on the MPAS mesh (``nCells`` dimension) in millimetres, with ``lonCell``/``latCell`` coordinates attached.
    """
    # Extract the set of available variables in the diagnostic file
    with xr.open_dataset(diag_file) as probe:
        available_vars = set(probe.data_vars)

    # Identify which precipitation components are available in the file, prefer both if present 
    precip_components = [v for v in ("rainc", "rainnc") if v in available_vars]

    # Ensure at least one precipitation component is available; otherwise, raise a ValueError.
    if not precip_components:
        raise ValueError(
            f"No 'rainc' or 'rainnc' variable in {diag_file}"
        )

    # Load the dataset with only precipitation variables, dropping all others to save memory. 
    dataset = xr.load_dataset(diag_file, drop_variables=[
        v for v in available_vars if v not in precip_components
    ])

    if "lonCell" not in dataset and "latCell" not in dataset:
        # If the dataset file doesn't contain the necessary coordinates, load them from the grid file 
        grid_dataset = xr.load_dataset(
            grid_file, drop_variables=_variables_to_drop(grid_file, ("lonCell", "latCell")),
        )

        # Extract and merge grid coordinates into the dataset for remapping. 
        dataset["lonCell"] = grid_dataset["lonCell"]
        dataset["latCell"] = grid_dataset["latCell"]

        # Close grid dataset to free memory
        grid_dataset.close()

    # Compute total precipitation
    if "rainc" in dataset and "rainnc" in dataset:
        # Compute total precipitation from rainc and rainnc
        total_precip = dataset["rainc"].isel(Time=0) + dataset["rainnc"].isel(Time=0)
    elif "rainc" in dataset:
        # Compute total precipitation from rainc
        total_precip = dataset["rainc"].isel(Time=0)
    else:
        # Compute total precipitation from rainnc
        total_precip = dataset["rainnc"].isel(Time=0)

    # Update variable attribute
    total_precip.attrs["units"] = "mm"
    total_precip.attrs["_mpas_dataset_path"] = diag_file

    # Close the original dataset to free memory
    dataset.close()

    # Remove the original dataset reference to free memory
    del dataset

    # Return the total precipitation DataArray on the MPAS mesh with coordinates for remapping.
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
        data (xr.DataArray): Precipitation or other field on the unstructured MPAS mesh (``nCells`` dimension).
        dataset_or_file (str): Path to the MPAS dataset file providing coordinate variables for bounds extraction.
        grid_file (str): Path to the MPAS grid file containing ``lonCell``/``latCell`` if not in the dataset file.
        resolution (float): Target grid spacing in degrees (default: 0.1°).

    Returns:
        xr.DataArray: Remapped field on a regular lat-lon grid with ``latitude`` and ``longitude`` coordinates.
    """
    # Ensure mpasdiag is available before attempting to load or remap data
    _ensure_mpasdiag_available()

    # Load necessary MPASdiag modules
    from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking  # type: ignore[import-untyped]
    from mpasdiag.processing.utils_geog import MPASGeographicUtils  # type: ignore[import-untyped]

    # Only load coordinate variables needed for geographic bounds extraction
    coord_vars_to_keep = ("lonCell", "latCell")

    # Load the dataset to access coordinates for remapping, dropping unnecessary variables to save memory.
    dataset = xr.load_dataset(
        dataset_or_file, drop_variables=_variables_to_drop(dataset_or_file, coord_vars_to_keep),
    )

    if "lonCell" not in dataset:
        # If the dataset file doesn't contain the necessary coordinates, load them from the grid file 
        grid_dataset = xr.load_dataset(
            grid_file, drop_variables=_variables_to_drop(grid_file, coord_vars_to_keep),
        )

        # Extract and merge grid coordinates into the dataset for remapping. 
        dataset["lonCell"] = grid_dataset["lonCell"]
        dataset["latCell"] = grid_dataset["latCell"]

        # Close the grid dataset immediately after merging coordinates to free resources
        grid_dataset.close()

    # Extract geographic bounds from the grid
    lon, lat = MPASGeographicUtils.extract_spatial_coordinates(dataset, normalize=False)

    # Extract the geographic extent of the MPAS mesh with no buffer since remap_mpas_to_latlon_with_masking will handle out-of-domain points via masking.
    lon_min, lon_max, lat_min, lat_max = (
        MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=0.0)
    )

    # Log the determined geographic extent and target resolution
    logger.debug(
        "MPAS remapping: extent [%.2f, %.2f] x [%.2f, %.2f], res=%.3f°",
        lon_min, lon_max, lat_min, lat_max, resolution,
    )

    # Remap the data to a regular lat-lon grid with masking of out-of-domain points.
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

    # Close the dataset after remapping to free resources
    dataset.close()

    # Standardize latitude dimension name
    if "lat" in remapped_da.dims and "latitude" not in remapped_da.dims:
        remapped_da = remapped_da.rename({"lat": "latitude"})

    # Standardize longitude dimension name  
    if "lon" in remapped_da.dims and "longitude" not in remapped_da.dims:
        remapped_da = remapped_da.rename({"lon": "longitude"})

    # Return the remapped DataArray on a regular lat-lon grid 
    return remapped_da


def load_and_remap_mpas_precip(
    diag_file: str,
    grid_file: str,
    resolution: float = 0.1,
) -> xr.DataArray:
    """
    Load total precipitation from an MPAS diag file and remap it to a regular lat-lon grid. This is the primary public entry point for MPAS forecast data, combining load_mpas_precip and remap_to_latlon into a single convenience call. It is used by FileManager.accumulate_forecasts when processing individual MPAS time steps. The mpasdiag library is required and will be lazily imported on the first call via ``_ensure_mpasdiag_available``.

    Parameters:
        diag_file (str): Path to an MPAS ``diag.*`` NetCDF file containing precipitation variables.
        grid_file (str): Path to the MPAS grid file for coordinate and bounds information.
        resolution (float): Target grid spacing for the output lat-lon grid in degrees (default: 0.1°).

    Returns:
        xr.DataArray: Total precipitation on a regular lat-lon grid in millimetres.
    """
    # Ensure mpasdiag is available before attempting to load or remap data
    _ensure_mpasdiag_available()

    # Load total precipitation on the MPAS mesh from the diagnostic file, merging grid coordinates as needed.
    precip_mesh = load_mpas_precip(diag_file, grid_file)

    # Return the remapped precipitation field on a regular lat-lon grid for verification against observations. 
    return remap_to_latlon(precip_mesh, diag_file, grid_file, resolution)
