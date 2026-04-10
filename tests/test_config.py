#!/usr/bin/env python3

"""
Unit tests for MODvx configuration management.

This module contains tests for the ModvxConfig class and its associated YAML loading function. The tests verify that configuration parameters are correctly parsed, default values are applied, and error handling works as expected for invalid inputs. By ensuring the integrity of the configuration system, these tests help maintain the robustness and reliability of the overall verification workflow.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

import datetime
from pathlib import Path

import pytest

from modvx.config import (
    ModvxConfig,
    _parse_datetime_string,
    load_config_from_yaml,
    apply_cli_overrides,
)


class TestModvxConfig:
    """ Tests for the ModvxConfig dataclass, which encapsulates all configuration parameters for the MODvx verification workflow. """

    def test_defaults(self: "TestModvxConfig") -> None:
        """
        This test confirms that the default values for ModvxConfig fields are set correctly. It instantiates a ModvxConfig with no arguments and asserts that the default values for experiment_name, thresholds, window_sizes, and forecast_step match the expected defaults. This ensures that users who rely on default configuration settings will have a consistent starting point without needing to specify every parameter.

        Parameters:
            None

        Returns:
            None
        """
        cfg = ModvxConfig()
        assert cfg.experiment_name == "liuz_coldstart_15km2025"
        assert cfg.thresholds == [90.0, 95.0, 97.5, 99.0]
        assert cfg.window_sizes == [1, 3, 5, 7, 9, 11, 13, 15]
        assert cfg.forecast_step == datetime.timedelta(hours=12)

    def test_resolve_relative_path(self: "TestModvxConfig") -> None:
        """
        This test verifies that the resolve_relative_path method correctly resolves a relative path against the base_dir. It creates a ModvxConfig instance with a specified base_dir and calls resolve_relative_path with a relative path. The test asserts that the returned path is the correct combination of base_dir and the relative path, ensuring that file paths are constructed correctly for output directories and other file-based configuration parameters.

        Parameters:
            None

        Returns:
            None
        """
        cfg = ModvxConfig(base_dir="/data")
        assert cfg.resolve_relative_path("output") == "/data/output"

    def test_timedelta_properties(self: "TestModvxConfig") -> None:
        """
        This test confirms that the forecast_step and forecast_length properties correctly convert their respective hour fields to datetime.timedelta objects. It creates a ModvxConfig instance with specific values for forecast_step_hours and forecast_length_hours, then asserts that the properties return the expected timedelta objects. This ensures that any code relying on these properties receives the correct time intervals for forecasting and verification. 

        Parameters:
            None

        Returns:
            None
        """
        cfg = ModvxConfig(
            forecast_step_hours=6,
            observation_interval_hours=1,
            cycle_interval_hours=12,
            forecast_length_hours=72,
        )
        assert cfg.forecast_step == datetime.timedelta(hours=6)
        assert cfg.forecast_length == datetime.timedelta(hours=72)


class TestLoadConfig:
    """ Tests for the load_config_from_yaml function, which parses a YAML file into a ModvxConfig instance. """

    def test_load_simple_yaml(self: "TestLoadConfig", 
                              tmp_path: Path) -> None:
        """
        This test verifies that load_config_from_yaml correctly parses a simple YAML configuration file into a ModvxConfig instance. It creates a temporary YAML file with known configuration values, loads it using the function, and asserts that the resulting ModvxConfig object has the expected field values. This ensures that the YAML parsing logic correctly maps YAML keys to ModvxConfig fields and handles basic data types as intended. 

        Parameters:
            tmp_path (Path): Pytest-provided temporary directory, isolated per test invocation.

        Returns:
            None
        """
        yaml_text = """\
experiment_name: "test_exp"
forecast_step_hours: 6
thresholds:
  - 90.0
  - 99.0
vxdomain:
  - TROPICS
initial_cycle_start: "20240101T00"
final_cycle_start: "20240110T00"
"""
        f = tmp_path / "test.yaml"
        f.write_text(yaml_text)
        cfg = load_config_from_yaml(f)

        assert cfg.experiment_name == "test_exp"
        assert cfg.forecast_step_hours == 6
        assert cfg.thresholds == [90.0, 99.0]
        assert cfg.vxdomain == ["TROPICS"]
        assert cfg.initial_cycle_start == datetime.datetime(2024, 1, 1, 0, 0, 0)

    def test_missing_file_raises(self: "TestLoadConfig") -> None:
        """
        This test ensures that load_config_from_yaml raises a FileNotFoundError when the specified YAML file does not exist. It calls the function with a clearly invalid file path and asserts that the expected exception is raised, confirming that the function properly handles missing files and provides appropriate feedback to users when their configuration file cannot be found.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(FileNotFoundError):
            load_config_from_yaml("/nonexistent/path.yaml")


class TestMergeCLI:
    """ Tests for the apply_cli_overrides function, which merges a dictionary of CLI overrides into a base ModvxConfig instance. """

    def test_override_experiment(self: "TestMergeCLI") -> None:
        """
        This test verifies that the apply_cli_overrides function correctly overrides the experiment_name field in a ModvxConfig instance when a new value is provided in the CLI overrides dictionary. It creates a base ModvxConfig with a specific experiment name, applies an override with a different name, and asserts that the resulting merged configuration reflects the new experiment name while the original base configuration remains unchanged. This confirms that the override mechanism works as intended and does not mutate the original configuration object. 

        Parameters:
            None

        Returns:
            None
        """
        base = ModvxConfig(experiment_name="old")
        merged = apply_cli_overrides(base, {"experiment_name": "new"})
        assert merged.experiment_name == "new"
        # Original unchanged
        assert base.experiment_name == "old"

    def test_none_values_ignored(self: "TestMergeCLI") -> None:
        """
        This test confirms that when a CLI override value is set to None, it does not override the existing value in the base ModvxConfig. This allows users to specify overrides without unintentionally nullifying existing configuration values. The test creates a base configuration with a known experiment name, applies an override with None for that field, and asserts that the original experiment name is retained in the merged configuration. It also checks that other valid overrides (like verbose=True) are still applied correctly. 

        Parameters:
            None

        Returns:
            None
        """
        base = ModvxConfig(experiment_name="keep_me")
        merged = apply_cli_overrides(base, {"experiment_name": None, "verbose": True})
        assert merged.experiment_name == "keep_me"
        assert merged.verbose is True

    def test_unknown_keys_ignored(self: "TestMergeCLI") -> None:
        """
        This test verifies that when the CLI overrides dictionary contains keys that do not correspond to any fields in ModvxConfig, those keys are ignored and do not affect the resulting configuration. This ensures that users can provide extra parameters in their CLI input without causing errors or unintended side effects on the configuration. The test creates a base ModvxConfig, applies an override with an unknown key, and asserts that the known fields (like experiment_name) remain unchanged while no exceptions are raised due to the unrecognized key. 

        Parameters:
            None

        Returns:
            None
        """
        base = ModvxConfig()
        merged = apply_cli_overrides(base, {"no_such_field": 42})
        assert merged.experiment_name == base.experiment_name


class TestConfigProperties:
    """ Tests for computed properties in ModvxConfig that convert integer hour fields to datetime.timedelta objects. """

    def test_observation_interval_property(self: "TestConfigProperties") -> None:
        """
        This test verifies that the observation_interval property correctly converts an integer hour value to a timedelta. This test supplies a custom three-hour observation interval and asserts that the property returns the matching datetime.timedelta object. Accurate observation interval conversion ensures that any code relying on this property receives the correct time interval for observations, which is critical for generating accurate forecast cycles and aligning verification data. 

        Parameters:
            None

        Returns:
            None
        """
        cfg = ModvxConfig(observation_interval_hours=3)
        assert cfg.observation_interval == datetime.timedelta(hours=3)

    def test_cycle_interval_property(self: "TestConfigProperties") -> None:
        """
        This test verifies that the cycle_interval property correctly converts an integer hour value to a timedelta. It supplies a custom twelve-hour cycle interval and asserts that the property returns the corresponding datetime.timedelta object. Proper cycle interval conversion is essential for ensuring that forecast cycles are generated at the correct intervals, which affects the timing of data processing and verification steps in the workflow. 

        Parameters:
            None

        Returns:
            None
        """
        cfg = ModvxConfig(cycle_interval_hours=12)
        assert cfg.cycle_interval == datetime.timedelta(hours=12)

    def test_parse_datetime_iso_full(self: "TestConfigProperties") -> None:
        """
        This test verifies that the _parse_datetime_str function correctly parses a full ISO 8601 datetime string with a 'T' separator. It provides a complete datetime string including date and time components, and asserts that the resulting datetime object matches the expected values. This ensures that the parser can handle standard ISO 8601 formats commonly used in configuration files for specifying cycle start times and other datetime parameters. 

        Parameters:
            None

        Returns:
            None
        """
        result = _parse_datetime_string("2024-09-17T00:00:00")
        assert result == datetime.datetime(2024, 9, 17, 0, 0, 0)

    def test_parse_datetime_space_format(self: "TestConfigProperties") -> None:
        """
        This test verifies that the _parse_datetime_str function can also parse a datetime string with a space separator instead of 'T'. It provides a datetime string in the format "YYYY-MM-DD HH:MM:SS" and asserts that the resulting datetime object matches the expected values. This test confirms that the parser is flexible enough to handle common variations in datetime formatting that users might include in their configuration files, improving usability and reducing potential parsing errors due to format differences. 

        Parameters:
            None

        Returns:
            None
        """
        result = _parse_datetime_string("2024-09-17 12:30:00")
        assert result == datetime.datetime(2024, 9, 17, 12, 30, 0)

    def test_parse_datetime_bad_raises(self: "TestConfigProperties") -> None:
        """
        This test confirms that the _parse_datetime_str function raises a ValueError when given an invalid datetime string that cannot be parsed. It provides a clearly non-datetime string and asserts that the expected exception is raised with a message indicating that the datetime could not be parsed. This ensures that the function properly handles invalid input and provides informative error messages to users when their configuration contains incorrectly formatted datetime strings. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            _parse_datetime_string("not-a-date")


class TestPrecipAccumConfig:
    """ Tests for the precip_accum_hours configuration parameter and its related computed properties in ModvxConfig. """

    def test_default_zero(self: "TestPrecipAccumConfig") -> None:
        """
        This test verifies that the default value for precip_accum_hours is set to 0 when a ModvxConfig instance is created without specifying this parameter. This confirms that the configuration system initializes the accumulation hours to a known default state, which is important for ensuring consistent behavior in downstream code that relies on this parameter to determine precipitation accumulation periods. 

        Parameters:
             None

        Returns:
            None
        """
        cfg = ModvxConfig()
        assert cfg.precip_accum_hours == 0

    def test_effective_falls_back_to_forecast_step(self: "TestPrecipAccumConfig") -> None:
        """
        This test confirms that when precip_accum_hours is set to 0, the effective_precip_accum_hours property correctly falls back to the value of forecast_step_hours. This ensures that in the absence of an explicit accumulation period, the system defaults to using the forecast step duration for precipitation accumulation, which maintains logical consistency in how accumulation periods are determined based on the forecast configuration. 

        Parameters:
            None

        Returns:
            None
        """
        cfg = ModvxConfig(forecast_step_hours=12, precip_accum_hours=0)
        assert cfg.effective_precip_accum_hours == 12

    def test_effective_uses_explicit_value(self: "TestPrecipAccumConfig") -> None:
        """
        This test verifies that when precip_accum_hours is set to a non-zero value, the effective_precip_accum_hours property returns that explicit value instead of falling back to forecast_step_hours. This confirms that the configuration system correctly prioritizes user-specified accumulation hours over the default fallback mechanism, allowing for flexible configuration of precipitation accumulation periods based on user preferences. 

        Parameters:
            None

        Returns:
            None
        """
        cfg = ModvxConfig(forecast_step_hours=1, precip_accum_hours=3)
        assert cfg.effective_precip_accum_hours == 3

    def test_precip_accum_timedelta_fallback(self: "TestPrecipAccumConfig") -> None:
        """
        This test verifies that the precip_accum property returns a timedelta corresponding to the forecast_step_hours when precip_accum_hours is set to 0. This ensures that when no explicit accumulation period is defined, the system correctly uses the forecast step duration as the accumulation period, which is critical for ensuring that precipitation accumulation calculations are based on a valid time interval even in the absence of user-specified accumulation hours. 

        Parameters:
            None

        Returns:
            None
        """
        cfg = ModvxConfig(forecast_step_hours=6, precip_accum_hours=0)
        assert cfg.precip_accum == datetime.timedelta(hours=6)

    def test_precip_accum_timedelta_explicit(self: "TestPrecipAccumConfig") -> None:
        """
        This test confirms that the precip_accum property returns a timedelta corresponding to the explicitly set precip_accum_hours when it is non-zero. This ensures that when a user specifies a particular accumulation period in hours, the system correctly converts that value to a timedelta for use in precipitation accumulation calculations, allowing for accurate time-based computations based on user-defined accumulation periods. 

        Parameters:
            None

        Returns:
            None
        """
        cfg = ModvxConfig(forecast_step_hours=1, precip_accum_hours=3)
        assert cfg.precip_accum == datetime.timedelta(hours=3)

    def test_yaml_round_trip(self: "TestPrecipAccumConfig", 
                             tmp_path: Path) -> None:
        """
        This test verifies that when a ModvxConfig is loaded from a YAML file with a specified precip_accum_hours value, the effective_precip_accum_hours property correctly reflects that value. It creates a temporary YAML file with a known precip_accum_hours setting, loads it into a ModvxConfig instance, and asserts that both precip_accum_hours and effective_precip_accum_hours return the expected values. This confirms that the YAML loading process correctly populates the configuration fields and that the computed properties behave as expected based on the loaded configuration. 

        Parameters:
            tmp_path (Path): Temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        yaml_text = """\
experiment_name: "accum_test"
forecast_step_hours: 1
observation_interval_hours: 1
precip_accum_hours: 6
"""
        f = tmp_path / "accum.yaml"
        f.write_text(yaml_text)
        cfg = load_config_from_yaml(f)
        assert cfg.precip_accum_hours == 6
        assert cfg.effective_precip_accum_hours == 6

    def test_cli_override(self: "TestPrecipAccumConfig") -> None:
        """
        This test confirms that the apply_cli_overrides function can successfully override the precip_accum_hours field in a ModvxConfig instance when a new value is provided in the CLI overrides dictionary. It creates a base ModvxConfig with a specific precip_accum_hours value, applies an override with a different value, and asserts that the resulting merged configuration reflects the new precip_accum_hours while the original base configuration remains unchanged. This ensures that users can effectively override precipitation accumulation settings via CLI inputs without affecting the original configuration object. 

        Parameters:
            None

        Returns:
            None
        """
        base = ModvxConfig(precip_accum_hours=0)
        merged = apply_cli_overrides(base, {"precip_accum_hours": 3})
        assert merged.precip_accum_hours == 3
        assert base.precip_accum_hours == 0
