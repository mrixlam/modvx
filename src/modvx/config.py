"""
Configuration management for modvx.

Provides a ``ModvxConfig`` dataclass that holds every tuneable parameter
and a ``load_config`` helper that reads a YAML file and returns a validated
instance.  CLI arguments can selectively override individual fields via
``merge_cli_overrides``.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Default configuration values (mirrored in configs/default.yaml)
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: List[float] = [90.0, 95.0, 97.5, 99.0]
_DEFAULT_WINDOWS: List[int] = [1, 3, 5, 7, 9, 11, 13, 15]
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
    Central configuration dataclass for a complete modvx FSS verification run.
    All path fields are stored as relative strings and resolved against ``base_dir``
    at runtime, so the same YAML file remains portable across machines and working
    directories. Temporal parameters are stored as raw integer hour counts and exposed
    as ``datetime.timedelta`` properties for convenient arithmetic. Default values
    reflect a typical global 48-hour forecast experiment configuration.
    """

    # ---- experiment / time range ------------------------------------------
    experiment_name: str = "liuz_coldstart_15km2025"
    initial_cycle_start: datetime.datetime = field(
        default_factory=lambda: datetime.datetime(2025, 6, 13, 0, 0, 0)
    )
    final_cycle_start: datetime.datetime = field(
        default_factory=lambda: datetime.datetime(2025, 7, 9, 0, 0, 0)
    )
    forecast_step_hours: int = 12
    observation_interval_hours: int = 1
    cycle_interval_hours: int = 24
    forecast_length_hours: int = 48

    # ---- MPAS mesh settings -----------------------------------------------
    mpas_grid_file: str = ""  # path to MPAS grid file (relative to base_dir)
    mpas_remap_resolution: float = 0.1  # degrees for MPAS-to-latlon remapping

    # ---- grid / resolution ------------------------------------------------
    target_resolution: Union[str, float] = "obs"

    # ---- verification domains ---------------------------------------------
    vxdomain: List[str] = field(default_factory=lambda: ["GLOBAL"])
    regions: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_REGIONS))

    # ---- metric parameters ------------------------------------------------
    thresholds: List[float] = field(default_factory=lambda: list(_DEFAULT_THRESHOLDS))
    window_sizes: List[int] = field(default_factory=lambda: list(_DEFAULT_WINDOWS))
    threshold_mode: str = "independent"  # "independent" | "obs_only"

    # ---- variable names ---------------------------------------------------
    obs_var_name: str = "precip"

    # ---- directories ------------------------------------------------------
    base_dir: str = "."
    fcst_dir: str = "fcst"
    obs_dir: str = "obs/FIMERG"
    mask_dir: str = "masks"
    output_dir: str = "output"
    debug_dir: str = "debug"
    log_dir: str = "logs"
    csv_dir: str = "fss_csv"
    plot_dir: str = "fss_plots"
    cache_dir: Optional[str] = None

    # ---- filename templates -----------------------------------------------
    observation_template: str = (
        "{obs_dir}/IMERG.A01H.VLD{date_key}.S{date_key}T000000."
        "E{date_key}T235959.{vintage}.V07B.SRCHHR.X3600Y1800.R0p1.FMT.nc"
    )
    obs_vintage_preference: List[str] = field(
        default_factory=lambda: ["FNL", "LTE"]
    )

    # ---- I/O tuning -------------------------------------------------------
    compression_level: int = 9
    clip_buffer_deg: float = 1.0

    # ---- runtime flags ----------------------------------------------------
    verbose: bool = False
    save_intermediate: bool = False
    enable_logs: bool = False

    # ---- convenience helpers ----------------------------------------------

    @property
    def forecast_step(self) -> datetime.timedelta:
        return datetime.timedelta(hours=self.forecast_step_hours)

    @property
    def observation_interval(self) -> datetime.timedelta:
        return datetime.timedelta(hours=self.observation_interval_hours)

    @property
    def cycle_interval(self) -> datetime.timedelta:
        return datetime.timedelta(hours=self.cycle_interval_hours)

    @property
    def forecast_length(self) -> datetime.timedelta:
        return datetime.timedelta(hours=self.forecast_length_hours)

    def resolve_path(self, rel: str) -> str:
        """
        Resolve a relative path string against the configured base directory.
        All directory fields in ModvxConfig are stored as relative strings and must
        be resolved before use in filesystem operations. This helper centralises that
        resolution so that the same YAML configuration file works regardless of the
        current working directory. The result is returned as a plain string for
        compatibility with os.path and xarray file-loading functions.

        Parameters:
            rel (str): Relative path string to resolve against ``base_dir``.

        Returns:
            str: Absolute path string formed by joining ``base_dir`` and *rel*.
        """
        return str(Path(self.base_dir) / rel)

    def resolve_mask_path(self, mask_filename: str) -> str:
        """
        Resolve a mask filename to its full path under the configured mask directory.
        Mask files are stored in a dedicated subdirectory (``mask_dir``) beneath ``base_dir``.
        This helper constructs the full path by joining both directory levels with the filename.
        It is used by TaskManager when loading region masks specified in the regions dictionary.
        The result is a plain string suitable for passing to xarray or os.path functions.

        Parameters:
            mask_filename (str): Bare filename of the mask NetCDF file
                (e.g., ``"G004_GLOBAL.nc"``).

        Returns:
            str: Full path to the mask file under ``base_dir / mask_dir``.
        """
        return str(Path(self.base_dir) / self.mask_dir / mask_filename)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _parse_datetime_str(s: str) -> datetime.datetime:
    """
    Parse a compact or ISO-8601 datetime string into a Python datetime object.
    Accepted formats include the YAML-friendly ``yyyymmddThh`` compact form and
    standard ISO-8601 variants with full time components. The function tries each
    format sequentially and returns the first successful parse. A ValueError is
    raised with the offending string when none of the formats match.

    Parameters:
        s (str): Datetime string in ``yyyymmddThh`` or ISO-8601 format.

    Returns:
        datetime.datetime: Parsed datetime object.
    """
    for fmt in ("%Y%m%dT%H", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {s!r}")


def _coerce_value(key: str, raw: Any) -> Any:
    """
    Coerce a raw YAML scalar value to the Python type expected by ModvxConfig.
    YAML loaders return string values as plain Python strings, but certain ModvxConfig
    fields require datetime objects. This function identifies those fields by name
    and applies the appropriate conversion via _parse_datetime_str. All other fields
    are returned unchanged, relying on the dataclass constructor to perform any
    remaining type coercion.

    Parameters:
        key (str): ModvxConfig field name corresponding to the YAML key.
        raw (Any): Raw value as returned by the YAML loader.

    Returns:
        Any: Coerced value with the appropriate Python type for the given field.
    """
    dt_keys = {"initial_cycle_start", "final_cycle_start"}
    if key in dt_keys and isinstance(raw, str):
        return _parse_datetime_str(raw)
    return raw


def load_config(yaml_path: Union[str, Path]) -> ModvxConfig:
    """
    Load a YAML configuration file and return a fully populated ModvxConfig instance.
    The function reads the YAML file, coerces each recognised field to its expected Python
    type via _coerce_value, and constructs a ModvxConfig dataclass with the merged values.
    Unknown YAML keys are silently ignored so that user configuration files can include
    comments or extra entries without raising errors. A FileNotFoundError is raised
    immediately when the specified path does not exist.

    Parameters:
        yaml_path (str or Path): Path to the YAML configuration file.

    Returns:
        ModvxConfig: Validated configuration object populated from the YAML file.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    kwargs: Dict[str, Any] = {}
    valid_fields = {f.name for f in ModvxConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    for key, val in raw.items():
        if key in valid_fields:
            kwargs[key] = _coerce_value(key, val)

    return ModvxConfig(**kwargs)


def merge_cli_overrides(config: ModvxConfig, overrides: Dict[str, Any]) -> ModvxConfig:
    """
    Create and return a new ModvxConfig with selected fields overridden by CLI-provided values.
    This function is non-destructive — it copies all fields from the base configuration and
    applies only the non-``None`` entries from *overrides*, leaving everything else unchanged.
    Unknown keys are silently ignored so that argparse Namespace objects can be passed directly
    without prior filtering. Type coercion is applied to datetime fields via _coerce_value.

    Parameters:
        config (ModvxConfig): Base configuration, typically loaded from a YAML file.
        overrides (dict): Mapping of ModvxConfig field names to override values;
            ``None`` values are skipped.

    Returns:
        ModvxConfig: New configuration instance with the specified overrides applied.
    """
    valid_fields = {f.name for f in ModvxConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    merged: Dict[str, Any] = {}
    for f in ModvxConfig.__dataclass_fields__.values():  # type: ignore[attr-defined]
        merged[f.name] = getattr(config, f.name)
    for key, val in overrides.items():
        if val is not None and key in valid_fields:
            merged[key] = _coerce_value(key, val)
    return ModvxConfig(**merged)
