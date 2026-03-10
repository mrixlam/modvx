"""Tests for modvx.config."""

import datetime
from pathlib import Path

import pytest

from modvx.config import ModvxConfig, load_config, merge_cli_overrides


class TestModvxConfig:
    """Unit tests for the ModvxConfig dataclass and its computed helper methods."""

    def test_defaults(self) -> None:
        """Verify that ModvxConfig initializes with the expected default values. This test confirms that the dataclass provides sensible defaults without requiring explicit configuration arguments. It checks the experiment name string, the threshold float list, the window sizes integer list, and the computed forecast_step timedelta property. Passing this test ensures the dataclass factory defaults are correctly declared and consistently applied."""
        cfg = ModvxConfig()
        assert cfg.experiment_name == "liuz_coldstart_15km2025"
        assert cfg.thresholds == [90.0, 95.0, 97.5, 99.0]
        assert cfg.window_sizes == [1, 3, 5, 7, 9, 11, 13, 15]
        assert cfg.forecast_step == datetime.timedelta(hours=12)

    def test_resolve_path(self) -> None:
        """Confirm that resolve_path correctly joins the base directory with a relative subpath. This test instantiates ModvxConfig with a known base_dir value and asserts that the resolved path matches the expected absolute string. It serves as a minimal contract check for the path construction logic used throughout the pipeline when building forecast and output directories."""
        cfg = ModvxConfig(base_dir="/data")
        assert cfg.resolve_path("output") == "/data/output"

    def test_timedelta_properties(self) -> None:
        """Verify that the computed timedelta properties correctly reflect custom integer hour inputs. This test constructs a ModvxConfig with non-default hour values for forecast step, observation interval, cycle interval, and forecast length. It then asserts that the forecast_step and forecast_length properties return the correct datetime.timedelta objects. This ensures the property-based conversion from integer hours to timedeltas behaves correctly under non-default configuration."""
        cfg = ModvxConfig(
            forecast_step_hours=6,
            observation_interval_hours=1,
            cycle_interval_hours=12,
            forecast_length_hours=72,
        )
        assert cfg.forecast_step == datetime.timedelta(hours=6)
        assert cfg.forecast_length == datetime.timedelta(hours=72)


class TestLoadConfig:
    """Round-trip tests verifying YAML deserialization into ModvxConfig instances."""

    def test_load_simple_yaml(self, tmp_path: Path) -> None:
        """Write a minimal YAML file to a temporary directory and verify load_config parses it correctly. This test covers field coercion, including string-to-datetime parsing for cycle start fields and float list parsing for thresholds. It confirms the round-trip from YAML text to a ModvxConfig instance produces the expected field values. Using pytest's tmp_path fixture ensures filesystem isolation without requiring manual cleanup after the test."""
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
        cfg = load_config(f)

        assert cfg.experiment_name == "test_exp"
        assert cfg.forecast_step_hours == 6
        assert cfg.thresholds == [90.0, 99.0]
        assert cfg.vxdomain == ["TROPICS"]
        assert cfg.initial_cycle_start == datetime.datetime(2024, 1, 1, 0, 0, 0)

    def test_missing_file_raises(self) -> None:
        """Ensure load_config raises FileNotFoundError when given a path to a nonexistent file. This test guards against silent failures or misleading exceptions when a configuration file is absent at runtime. It verifies that the expected exception type is correctly propagated from the internal path resolution and file-open logic."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestMergeCLI:
    """Tests for CLI-level override merging into an existing ModvxConfig instance."""

    def test_override_experiment(self) -> None:
        """Confirm that merge_cli_overrides replaces a field value when the corresponding override is non-None. This test also verifies that the original ModvxConfig instance is not mutated after the merge, asserting immutability of the base config object. It exercises the most common CLI use case of overriding the experiment name from the command line."""
        base = ModvxConfig(experiment_name="old")
        merged = merge_cli_overrides(base, {"experiment_name": "new"})
        assert merged.experiment_name == "new"
        # Original unchanged
        assert base.experiment_name == "old"

    def test_none_values_ignored(self) -> None:
        """Verify that None values in the CLI override dictionary are skipped, leaving base field values intact. This guards against CLI arguments that default to None from accidentally overwriting existing config fields with null data. The test also confirms that non-None overrides present in the same dictionary are applied correctly alongside the skipped entries."""
        base = ModvxConfig(experiment_name="keep_me")
        merged = merge_cli_overrides(base, {"experiment_name": None, "verbose": True})
        assert merged.experiment_name == "keep_me"
        assert merged.verbose is True

    def test_unknown_keys_ignored(self) -> None:
        """Ensure that keys in the override dictionary that do not correspond to ModvxConfig fields are silently ignored. This protects against AttributeError or unexpected behavior when CLI parsers pass through extra or unrecognized arguments. It confirms that merge_cli_overrides is defensive about input key validity and does not alter the base config when no valid overrides are present."""
        base = ModvxConfig()
        merged = merge_cli_overrides(base, {"no_such_field": 42})
        assert merged.experiment_name == base.experiment_name
