#!/usr/bin/env python3

"""
Configuration management for MODvx.

This module defines the ModvxConfig dataclass, which encapsulates all configuration parameters for a MODvx verification run. It includes functionality to load configuration from a YAML file, apply command-line overrides, and provide convenient properties for time intervals. The configuration parameters cover experiment details, data paths, verification settings, and output options. This centralised configuration management allows for flexible and consistent handling of user inputs across the entire codebase. 

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
    "SINGV": "SINGV.nc",
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
    """ Central configuration dataclass for a complete modvx verification run. """

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
    mpas_remap_resolution: float = 1.0  

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

    # Specify the tag embedded in observation filenames after the vintage token
    obs_file_tag: str = "V07B.SRCHHR.X360Y180.R1p0.FMT"

    # Specify the template for constructing observation file paths
    observation_template: str = (
        "{obs_dir}/IMERG.A01H.VLD{date_key}.S{date_key}T000000."
        "E{date_key}T235959.{vintage}.{obs_file_tag}.nc"
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
    def forecast_step(self: "ModvxConfig") -> datetime.timedelta:
        """ 
        This property returns the forecast step interval as a datetime.timedelta object, which is computed by converting the configured forecast_step_hours (an integer representing hours) into a timedelta. This allows for consistent handling of time intervals throughout the codebase, as all time-related parameters can be accessed in a standardized format. The forecast_step_hours field is expected to be set in the YAML configuration file and can be overridden by command-line arguments, but this property ensures that any access to the forecast step interval will always receive a properly formatted timedelta object regardless of how the underlying hours value was specified. 

        Parameters:
            None
        
        Returns:
            datetime.timedelta: The forecast step interval represented as a timedelta object, computed from the forecast_step_hours field.
        """
        # Return the forecast step interval as a timedelta by converting the forecast_step_hours to hours
        return datetime.timedelta(hours=self.forecast_step_hours)


    @property
    def effective_precip_accum_hours(self: "ModvxConfig") -> int:
        """ 
        This property computes the effective precipitation accumulation period in hours based on the configuration. If the user has explicitly set precip_accum_hours to a positive value, that value is returned directly as the accumulation period. However, if precip_accum_hours is set to 0 (the default), it indicates that the accumulation period should be the same as the forecast step interval, which is determined by forecast_step_hours. This allows for flexible configuration where users can either specify a custom accumulation period or simply use the forecast step interval as the default accumulation period without needing to set it explicitly. By centralizing this logic in a property, we ensure that all parts of the code that need to access the effective precipitation accumulation period can do so consistently and without needing to duplicate this conditional logic. 

        Parameters:
            None

        Returns:
            int: The effective precipitation accumulation period in hours.
        """
        # Return precip_accum_hours if set, otherwise fall back to forecast_step_hours
        return self.precip_accum_hours if self.precip_accum_hours > 0 else self.forecast_step_hours


    @property
    def precip_accum(self: "ModvxConfig") -> datetime.timedelta:
        """ 
        This property returns the effective precipitation accumulation period as a datetime.timedelta object, which is computed from the effective_precip_accum_hours property. This allows for consistent handling of time intervals throughout the codebase, as all time-related parameters can be accessed in a standardized format. The effective_precip_accum_hours property determines the accumulation period in hours based on the configuration, and this precip_accum property converts that value into a timedelta for use in any context where a timedelta is required (e.g., when calculating valid times for observations or forecasts). By providing this as a property, we ensure that any access to the precipitation accumulation period will always receive a properly formatted timedelta object regardless of how the underlying hours value was specified in the configuration. 

        Parameters:
            None

        Returns:
            datetime.timedelta: The effective precipitation accumulation period represented as a timedelta object, computed from the effective_precip_accum_hours field.
        """
        # Return the effective precipitation accumulation period as a timedelta
        return datetime.timedelta(hours=self.effective_precip_accum_hours)


    @property
    def observation_interval(self: "ModvxConfig") -> datetime.timedelta:
        """ 
        This property returns the observation interval as a datetime.timedelta object, which is computed by converting the configured observation_interval_hours (an integer representing hours) into a timedelta. This allows for consistent handling of time intervals throughout the codebase, as all time-related parameters can be accessed in a standardized format. The observation_interval_hours field is expected to be set in the YAML configuration file and can be overridden by command-line arguments, but this property ensures that any access to the observation interval will always receive a properly formatted timedelta object regardless of how the underlying hours value was specified. 

        Parameters:
            None

        Returns:
            datetime.timedelta: The observation interval represented as a timedelta object, computed from the observation_interval_hours field.
        """
        # Return the observation interval as a datetime.timedelta
        return datetime.timedelta(hours=self.observation_interval_hours)


    @property
    def cycle_interval(self: "ModvxConfig") -> datetime.timedelta:
        """ 
        This property returns the cycle interval as a datetime.timedelta object, which is computed by converting the configured cycle_interval_hours (an integer representing hours) into a timedelta. This allows for consistent handling of time intervals throughout the codebase, as all time-related parameters can be accessed in a standardized format. The cycle_interval_hours field is expected to be set in the YAML configuration file and can be overridden by command-line arguments, but this property ensures that any access to the cycle interval will always receive a properly formatted timedelta object regardless of how the underlying hours value was specified. 

        Parameters:
            None

        Returns:
            datetime.timedelta: The cycle interval represented as a timedelta object, computed from the cycle_interval_hours field.
        """
        # Return the cycle interval as a datetime.timedelta
        return datetime.timedelta(hours=self.cycle_interval_hours)


    @property
    def forecast_length(self: "ModvxConfig") -> datetime.timedelta:
        """ 
        This property returns the forecast length as a datetime.timedelta object, which is computed by converting the configured forecast_length_hours (an integer representing hours) into a timedelta. This allows for consistent handling of time intervals throughout the codebase, as all time-related parameters can be accessed in a standardized format. The forecast_length_hours field is expected to be set in the YAML configuration file and can be overridden by command-line arguments, but this property ensures that any access to the forecast length will always receive a properly formatted timedelta object regardless of how the underlying hours value was specified. 

        Parameters:
            None

        Returns:
            datetime.timedelta: The forecast length represented as a timedelta object, computed from the forecast_length_hours field.
        """
        # Return the forecast length as a datetime.timedelta
        return datetime.timedelta(hours=self.forecast_length_hours)


    def resolve_relative_path(self: "ModvxConfig", 
                              rel: str) -> str:
        """
        This helper method resolves a relative path string against the configured base directory. It takes a relative path (e.g., "fcst/data.nc") and returns the absolute path formed by joining the base_dir with the provided relative path. This is useful for constructing full paths to input data, output files, or intermediate results based on the structured directory layout defined in the configuration. By centralizing this path resolution logic in a method, we ensure that all parts of the code that need to access files can do so consistently and without needing to manually join paths or worry about the base directory. The result is returned as a plain string suitable for passing to file I/O functions. 

        Parameters:
            rel (str): Relative path string to resolve against ``base_dir``.

        Returns:
            str: Absolute path string formed by joining ``base_dir`` and *rel*.
        """
        # Return the resolved path by joining the base directory with the provided relative path
        return str(Path(self.base_dir) / rel)


    def resolve_mask_path(self: "ModvxConfig", 
                          mask_filename: str) -> str:
        """
        This helper method resolves the full path to a mask file based on the configured base directory and mask directory. It takes a bare filename of a mask NetCDF file (e.g., "G004_GLOBAL.nc") and returns the absolute path formed by joining the base_dir, mask_dir, and the provided filename. This is specifically designed for accessing region mask files that are stored in a dedicated masks directory under the base directory. By using this method, any part of the code that needs to access a mask file can simply provide the filename and rely on this method to construct the correct path, ensuring consistency and reducing the likelihood of path-related errors. The result is returned as a plain string suitable for passing to file I/O functions. 

        Parameters:
            mask_filename (str): Bare filename of the mask NetCDF file (e.g., ``"G004_GLOBAL.nc"``).

        Returns:
            str: Full path to the mask file under ``base_dir / mask_dir``.
        """
        # Return the resolved mask path by joining the base directory, mask directory, and the provided mask filename
        return str(Path(self.base_dir) / self.mask_dir / mask_filename)


def _parse_datetime_string(s: str) -> datetime.datetime:
    """
    This helper function attempts to parse a datetime string into a datetime.datetime object. It supports multiple common formats, including the compact "yyyymmddThh" format and standard ISO-8601 formats with or without seconds. The function iterates through a predefined list of accepted datetime formats and tries to parse the input string against each format until a successful parse is achieved. If the input string does not match any of the accepted formats, a ValueError is raised indicating that the datetime could not be parsed. This function is used to coerce string values from YAML configurations or CLI arguments into proper datetime objects for use in the ModvxConfig dataclass. 

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


def _coerce_config_value(key: str, 
                         raw: Any) -> Any:
    """
    This helper function coerces raw values from YAML configurations or CLI arguments into the appropriate Python types based on the ModvxConfig field they correspond to. It checks if the given key corresponds to a datetime field (e.g., "initial_cycle_start" or "final_cycle_start") and if the raw value is a string, it attempts to parse it into a datetime object using the _parse_datetime_string function. For all other fields, it returns the raw value unchanged, allowing the YAML loader to handle type coercion for basic types like integers, floats, lists, etc. This function centralizes the logic for type coercion of configuration values, ensuring that any datetime fields are consistently parsed regardless of how they were specified in the YAML file or CLI arguments. 

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
    This function loads a ModvxConfig configuration object from a YAML file. It takes the path to the YAML file as input, reads and parses the file using the PyYAML library, and then constructs a ModvxConfig instance by coercing the raw YAML values into the appropriate types. The function checks for the existence of the specified YAML file and raises a FileNotFoundError if it does not exist. It also uses the _coerce_config_value helper function to ensure that any datetime fields are properly parsed from strings. The resulting ModvxConfig object is fully populated with values from the YAML file, ready for use in the rest of the codebase. 

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


def apply_cli_overrides(config: ModvxConfig, 
                        overrides: Dict[str, Any]) -> ModvxConfig:
    """
    This function applies command-line overrides to a base ModvxConfig configuration object. It takes the original configuration and a dictionary of overrides (where keys are ModvxConfig field names and values are the override values from the CLI). The function creates a new dictionary that starts with all the fields from the original configuration and then updates it with any non-None override values for recognized fields, coercing types as needed using the _coerce_config_value helper function. Finally, it constructs and returns a new ModvxConfig instance using the merged values. This allows users to specify a base configuration in a YAML file and then selectively override specific parameters via command-line arguments without needing to modify the YAML file directly. 

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


