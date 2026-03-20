#!/usr/bin/env python3

"""
Configuration management for MODvx.

This module defines the ModvxConfig class, which encapsulates all configuration parameters for a modvx verification run. The configuration is typically loaded from a YAML file and may be overridden by command-line arguments. The ModvxConfig class provides structured access to all settings, including paths, forecast and observation parameters, verification domains, thresholds, and parallel processing options. By centralizing configuration management in this class, we can ensure consistent handling of parameters across the entire pipeline and provide a single source of truth for all configurable aspects of the verification workflow. The module also includes a helper function to load the configuration from a YAML file and apply any necessary overrides.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
from __future__ import annotations

import yaml
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Specify the default percentile thresholds for verification metric computation. 
_DEFAULT_THRESHOLDS: List[float] = [90.0, 95.0, 97.5, 99.0]

# Specify the default window sizes (in grid points) for FSS computation. 
_DEFAULT_WINDOWS: List[int] = [1, 3, 5, 7, 9, 11, 13, 15]

# Specify the default region masks available for verification. 
_DEFAULT_REGIONS: Dict[str, str] = {
    "SINGV": "singv_domain_mask.nc",
    "TROPICS": "G004_TROPICS.nc",
    "GLOBAL": "G004_GLOBAL.nc",
    "AFRICA": "G004_AFRICA.nc",
    "ASIA": "G004_ASIA.nc",
    "NAMERICA": "G004_NAMERICA.nc",
    "SAMERICA": "G004_SAMERICA.nc",
    "NHEM": "G004_NHEM.nc",
    "SHEM": "G004_SHEM.nc",
    "AUNZ": "WAFS0P25_AUNZ.nc",
}


@dataclass
class ModvxConfig:
    """
    Central configuration dataclass for a complete modvx FSS verification run. All path fields are stored as relative strings and resolved against ``base_dir`` at runtime, so the same YAML file remains portable across machines and working directories. Temporal parameters are stored as raw integer hour counts and exposed as ``datetime.timedelta`` properties for convenient arithmetic. Default values reflect a typical global 48-hour forecast experiment configuration.
    """
    # Set a default experiment name for use in output directories and file naming 
    experiment_name: str = "liuz_coldstart_15km2025"

    # Set the default initial cycle start date for the verification run
    initial_cycle_start: datetime.datetime = field(
        default_factory=lambda: datetime.datetime(2025, 6, 13, 0, 0, 0)
    )

    # Set the default final cycle start date for the verification run
    final_cycle_start: datetime.datetime = field(
        default_factory=lambda: datetime.datetime(2025, 7, 9, 0, 0, 0)
    )

    # Set the default forecast step interval in hours (e.g., 12 for 12-hourly forecasts)
    forecast_step_hours: int = 12

    # Set the default observation interval in hours (e.g., 1 for hourly observations)
    observation_interval_hours: int = 1

    # Set the default cycle interval in hours (e.g., 24 for daily cycles)
    cycle_interval_hours: int = 24

    # Set the default forecast length in hours (e.g., 48 for 48-hour forecasts)
    forecast_length_hours: int = 48

    # Set the precipitation accumulation period in hours (0 = use forecast_step_hours)
    precip_accum_hours: int = 0

    # Set the default path to the MPAS grid file (relative to base_dir)
    mpas_grid_file: str = ""  

    # Set the default resolution for remapping MPAS data to a lat-lon grid in degrees
    mpas_remap_resolution: float = 0.1  

    # Specify the target resolution for remapping MPAS data
    target_resolution: Union[str, float] = "obs"

    # Specify the verification domains to include in the run
    vxdomain: List[str] = field(default_factory=lambda: ["GLOBAL"])
    regions: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_REGIONS))

    # Specify the percentile thresholds and threshold mode used for computation
    thresholds: List[float] = field(default_factory=lambda: list(_DEFAULT_THRESHOLDS))
    threshold_mode: str = "independent"  # Options: "independent" or "obs_only"

    # Specify the window sizes (in grid points) for verification metric computation
    window_sizes: List[int] = field(default_factory=lambda: list(_DEFAULT_WINDOWS))

    # Specify the variable name used for verification
    obs_var_name: str = "precip"

    # Specify the directories for input data, output results, and intermediate files (relative to base_dir)
    base_dir: str = "."
    fcst_dir: str = "fcst"
    obs_dir: str = "obs/FIMERG"
    mask_dir: str = "masks"
    output_dir: str = "output"
    debug_dir: str = "debug"
    log_dir: str = "logs"
    csv_dir: str = "csv"
    plot_dir: str = "plots"
    cache_dir: Optional[str] = None

    # Specify the template for constructing observation file paths
    observation_template: str = (
        "{obs_dir}/IMERG.A01H.VLD{date_key}.S{date_key}T000000."
        "E{date_key}T235959.{vintage}.V07B.SRCHHR.X3600Y1800.R0p1.FMT.nc"
    )

    # Specify the preferred vintage of observation data to use when multiple are available
    obs_vintage_preference: List[str] = field(
        default_factory=lambda: ["FNL", "LTE"]
    )

    # Specify the compression level for output NetCDF files 
    compression_level: int = 9

    # Set the buffer distance in degrees for clipping forecasts to observation coverage areas.
    clip_buffer_deg: float = 1.0

    # Specify whether verbose logging is enabled 
    verbose: bool = False

    # Specify whether to save intermediate files during processing
    save_intermediate: bool = False

    # Specify whether to enable logging of debug information to a log file in log_dir
    enable_logs: bool = False


    @property
    def forecast_step(self) -> datetime.timedelta:
        # Return the forecast step interval as a datetime.timedelta 
        return datetime.timedelta(hours=self.forecast_step_hours)


    @property
    def effective_precip_accum_hours(self) -> int:
        # Return precip_accum_hours if set, otherwise fall back to forecast_step_hours
        return self.precip_accum_hours if self.precip_accum_hours > 0 else self.forecast_step_hours


    @property
    def precip_accum(self) -> datetime.timedelta:
        # Return the effective precipitation accumulation period as a timedelta
        return datetime.timedelta(hours=self.effective_precip_accum_hours)


    @property
    def observation_interval(self) -> datetime.timedelta:
        # Return the observation interval as a datetime.timedelta
        return datetime.timedelta(hours=self.observation_interval_hours)


    @property
    def cycle_interval(self) -> datetime.timedelta:
        # Return the cycle interval as a datetime.timedelta
        return datetime.timedelta(hours=self.cycle_interval_hours)


    @property
    def forecast_length(self) -> datetime.timedelta:
        # Return the forecast length as a datetime.timedelta
        return datetime.timedelta(hours=self.forecast_length_hours)


    def resolve_relative_path(self, rel: str) -> str:
        """
        Resolve a relative path string against the configured base directory. All directory fields in ModvxConfig are stored as relative strings and must be resolved before use in filesystem operations. This helper centralises that resolution so that the same YAML configuration file works regardless of the current working directory. The result is returned as a plain string for compatibility with os.path and xarray file-loading functions.

        Parameters:
            rel (str): Relative path string to resolve against ``base_dir``.

        Returns:
            str: Absolute path string formed by joining ``base_dir`` and *rel*.
        """
        # Return the resolved path by joining the base directory with the provided relative path
        return str(Path(self.base_dir) / rel)


    def resolve_mask_path(self, mask_filename: str) -> str:
        """
        Resolve a mask filename to its full path under the configured mask directory. Mask files are stored in a dedicated subdirectory (``mask_dir``) beneath ``base_dir``. This helper constructs the full path by joining both directory levels with the filename. It is used by TaskManager when loading region masks specified in the regions dictionary. The result is a plain string suitable for passing to xarray or os.path functions.

        Parameters:
            mask_filename (str): Bare filename of the mask NetCDF file (e.g., ``"G004_GLOBAL.nc"``).

        Returns:
            str: Full path to the mask file under ``base_dir / mask_dir``.
        """
        # Return the resolved mask path by joining the base directory, mask directory, and the provided mask filename
        return str(Path(self.base_dir) / self.mask_dir / mask_filename)


def _parse_datetime_string(s: str) -> datetime.datetime:
    """
    Parse a compact or ISO-8601 datetime string into a Python datetime object. Accepted formats include the YAML-friendly ``yyyymmddThh`` compact form and standard ISO-8601 variants with full time components. The function tries each format sequentially and returns the first successful parse. A ValueError is raised with the offending string when none of the formats match.

    Parameters:
        s (str): Datetime string in ``yyyymmddThh`` or ISO-8601 format.

    Returns:
        datetime.datetime: Parsed datetime object.
    """
    # Define the accepted datetime formats to try parsing against
    datetime_formats = ("%Y%m%dT%H", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")

    #Return the parsed datetime object if the string matches any of the accepted formats
    for fmt in datetime_formats:
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    
    # Raise a ValueError if the string does not match any of the accepted formats
    raise ValueError(f"Cannot parse datetime: {s!r}")


def _coerce_config_value(key: str, raw: Any) -> Any:
    """
    Coerce a raw YAML scalar value to the Python type expected by ModvxConfig. YAML loaders return string values as plain Python strings, but certain ModvxConfig fields require datetime objects. This function identifies those fields by name and applies the appropriate conversion via _parse_datetime_str. All other fields are returned unchanged, relying on the dataclass constructor to perform any remaining type coercion.

    Parameters:
        key (str): ModvxConfig field name corresponding to the YAML key.
        raw (Any): Raw value as returned by the YAML loader.

    Returns:
        Any: Coerced value with the appropriate Python type for the given field.
    """
    # Define the set of ModvxConfig fields that require datetime parsing
    dt_keys = {"initial_cycle_start", "final_cycle_start"}

    # Return the parsed datetime object if the key is in the set of datetime fields and the raw value is a string
    if key in dt_keys and isinstance(raw, str):
        return _parse_datetime_string(raw)
    
    # Return the raw value unchanged for all other fields
    return raw


def load_config_from_yaml(yaml_path: Union[str, Path]) -> ModvxConfig:
    """
    Load a YAML configuration file and return a fully populated ModvxConfig instance. The function reads the YAML file, coerces each recognised field to its expected Python type via _coerce_value, and constructs a ModvxConfig dataclass with the merged values. Unknown YAML keys are silently ignored so that user configuration files can include comments or extra entries without raising errors. A FileNotFoundError is raised immediately when the specified path does not exist.

    Parameters:
        yaml_path (str or Path): Path to the YAML configuration file.

    Returns:
        ModvxConfig: Validated configuration object populated from the YAML file.
    """
    # Convert the input path to a Path object if it is a string
    yaml_path = Path(yaml_path)

    # Raise a FileNotFoundError if the specified YAML configuration file does not exist at the given path
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    # Open the YAML file and load its contents into a raw dictionary
    with open(yaml_path, "r") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    # Start with an empty dictionary of keyword arguments
    kwargs: Dict[str, Any] = {}

    # Define the set of valid ModvxConfig field names for filtering the raw YAML keys
    valid_fields = {f.name for f in ModvxConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]

    # Iterate over the raw YAML key-value pairs and coerce values for recognized fields
    for key, val in raw.items():
        if key in valid_fields:
            kwargs[key] = _coerce_config_value(key, val)

    # Construct and return a ModvxConfig instance using the coerced keyword arguments
    return ModvxConfig(**kwargs)


def apply_cli_overrides(config: ModvxConfig, overrides: Dict[str, Any]) -> ModvxConfig:
    """
    Create and return a new ModvxConfig with selected fields overridden by CLI-provided values. This function is non-destructive — it copies all fields from the base configuration and applies only the non-``None`` entries from *overrides*, leaving everything else unchanged. Unknown keys are silently ignored so that argparse Namespace objects can be passed directly without prior filtering. Type coercion is applied to datetime fields via _coerce_value.

    Parameters:
        config (ModvxConfig): Base configuration, typically loaded from a YAML file.
        overrides (dict): Mapping of ModvxConfig field names to override values

    Returns:
        ModvxConfig: New configuration instance with the specified overrides applied.
    """
    # Define the set of valid ModvxConfig field names for filtering the overrides dictionary
    valid_fields = {f.name for f in ModvxConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]

    # Start with an empty dictionary to hold the merged configuration values
    merged: Dict[str, Any] = {}

    # First, copy all fields from the base configuration to the merged dictionary
    for f in ModvxConfig.__dataclass_fields__.values():  # type: ignore[attr-defined]
        merged[f.name] = getattr(config, f.name)
    
    # Then, iterate over the overrides and apply any non-None values for recognized fields, coercing types as needed
    for key, val in overrides.items():
        if val is not None and key in valid_fields:
            merged[key] = _coerce_config_value(key, val)
    
    # Construct and return a new ModvxConfig instance using the merged values
    return ModvxConfig(**merged)


