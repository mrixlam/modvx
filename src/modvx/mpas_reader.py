#!/usr/bin/env python3

"""
MPAS unstructured-mesh forecast reader for modvx.

This module provides functions to read total precipitation from MPAS diagnostic files on the native unstructured mesh and remap it to a regular latitude-longitude grid using the mpasdiag library. The primary public function is load_and_remap_mpas_precip, which combines loading and remapping into a single call for convenience. The module includes internal helper functions for efficient variable loading and checking for mpasdiag availability. The remapping process automatically determines geographic bounds from the MPAS mesh coordinates and applies nearest-neighbour interpolation with masking of out-of-domain points. The resulting remapped DataArray is compatible with the rest of the modvx pipeline for verification against observations.

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


def _variables_to_drop(filepath: str, 
                       keep: tuple[str, ...]) -> list[str]:
    """
    This internal helper function takes a file path to a NetCDF dataset and a tuple of variable names to keep, and returns a list of variable names that are present in the dataset but not in the keep tuple. This is used to identify which variables can be dropped when loading the dataset with xarray to save memory. The function opens the dataset in a context manager to read the variable names without loading the full data into memory, and constructs a list of variable names that should be dropped based on the provided keep tuple. 

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
    This internal helper function checks for the availability of the mpasdiag library, which is required for remapping MPAS data to a regular lat-lon grid. It uses a module-level cache variable to store the availability status after the first check, so that subsequent calls can return immediately without attempting to import again. If mpasdiag is not available, it raises an ImportError with instructions on how to install it. This function should be called at the beginning of any public function that relies on mpasdiag to ensure that the necessary dependencies are present before proceeding. 

    Parameters:
        None
        
    Returns:
        None
    """
    # Check the cached availability flag to avoid repeated import attempts
    global _HAS_MPASDIAG

    # If the availability has already been determined, return immediately
    if _HAS_MPASDIAG is True:
        return
    
    # If the availability has already been determined to be False, raise the same ImportError again
    try:
        # Import modules to verify availability; the imports themselves are used for checking availability
        import mpasdiag.processing.remapping   # type: ignore[import-untyped]  # noqa: F401
        import mpasdiag.processing.utils_geog  # type: ignore[import-untyped]  # noqa: F401
        _HAS_MPASDIAG = True
    except ImportError as exc:
        _HAS_MPASDIAG = False
        raise ImportError(
            "mpasdiag is required for MPAS mesh forecast reading. "
            "Install it with: pip install -e /path/to/MPASdiag"
        ) from exc


def load_mpas_precip(diag_file: str,
                     grid_file: str,) -> xr.DataArray:
    """
    This function loads total precipitation from an MPAS diagnostic file on the native unstructured mesh. It checks for the presence of ``rainc`` and/or ``rainnc`` variables in the diagnostic file, loads only those variables to save memory, and sums them if both are present to compute total precipitation. If the necessary coordinates for remapping (``lonCell``/``latCell``) are not present in the diagnostic file, it loads them from the provided grid file and merges them into the dataset. The resulting DataArray contains total cumulative precipitation on the MPAS mesh with appropriate attributes and coordinates for remapping. 

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


def remap_to_latlon(data: xr.DataArray,
                    dataset_or_file: str,
                    grid_file: str,
                    resolution: float = 0.1,) -> xr.DataArray:
    """
    This function remaps a DataArray defined on the unstructured MPAS mesh to a regular latitude-longitude grid using the mpasdiag library. It first ensures that mpasdiag is available, then loads only the necessary coordinate variables from the provided dataset or grid file to determine the geographic bounds of the MPAS mesh. The remapping is performed using nearest-neighbour interpolation with masking of out-of-domain points, and the resulting DataArray is returned with standardized latitude and longitude dimension names. This function is designed to be flexible in accepting either a dataset file that contains coordinates or a separate grid file, and it automatically handles the extraction of geographic bounds to ensure that the remapped grid covers the appropriate area. 

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


def load_and_remap_mpas_precip(diag_file: str,
                               grid_file: str,
                               resolution: float = 0.1,) -> xr.DataArray:
    """
    This function combines the loading of total precipitation from an MPAS diagnostic file on the native unstructured mesh and remapping it to a regular latitude-longitude grid in a single call for convenience. It first calls load_mpas_precip to read the precipitation field from the diagnostic file, then calls remap_to_latlon to remap it to a regular grid using the provided grid file for coordinate information. The resulting DataArray contains total precipitation on a regular lat-lon grid, ready for verification against observations or further analysis. 

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
