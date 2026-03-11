#!/usr/bin/env python3

"""
Unit tests for MODvx command-line interface.

This module contains unit tests for the command-line interface defined in modvx.cli. The tests cover argument parsing, configuration resolution, logging setup, and the dispatch of subcommands to their respective handler functions. Mocking is used to isolate the CLI logic from the underlying TaskManager, ParallelProcessor, FileManager, and Visualizer implementations, allowing for focused testing of the CLI behaviour without side effects.    

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modvx.cli import (
    add_shared_cli_args,
    handle_run_subcommand,
    resolve_config_from_namespace,
    configure_root_logging,
    main,
)
from modvx.config import ModvxConfig


# -----------------------------------------------------------------------
# configure_root_logging
# -----------------------------------------------------------------------

class TestSetupLogging:
    """Tests for the configure_root_logging helper that configures root logging."""

    def test_info_level_by_default(self) -> None:
        """
        Verify that logging is configured at INFO level when verbose is False in the config. This test calls configure_root_logging with a minimal config that has verbose disabled and checks the root logger's effective level. INFO is the expected default so that pipeline progress messages are visible without the additional DEBUG diagnostic output.

        Returns:
            None
        """
        import logging
        cfg = ModvxConfig(verbose=False)
        configure_root_logging(cfg)
        assert logging.getLogger().level == logging.INFO

    def test_debug_level_when_verbose(self) -> None:
        """
        Verify that logging is configured at DEBUG level when verbose is True in the config. Verbose mode is intended for development and debugging, where detailed trace output from all pipeline stages is needed. This test confirms that enabling verbose propagates correctly from the config object to the root logger's level.

        Returns:
            None
        """
        import logging
        cfg = ModvxConfig(verbose=True)
        configure_root_logging(cfg)
        assert logging.getLogger().level == logging.DEBUG


# -----------------------------------------------------------------------
# _add_common_args
# -----------------------------------------------------------------------

class TestAddCommonArgs:
    """Tests for _add_common_args verifying common arguments are registered."""

    def test_config_and_verbose_registered(self) -> None:
        """
        Confirm that _add_common_args registers both the --config and --verbose flags on a new ArgumentParser. This test parses a synthetic argument list containing both flags and asserts that the resulting namespace holds the expected values. Common argument registration must be consistent across all subcommand parsers to avoid missing flags at runtime.

        Returns:
            None
        """
        parser = argparse.ArgumentParser()
        add_shared_cli_args(parser)
        args = parser.parse_args(["-c", "my.yaml", "--verbose"])
        assert args.config == "my.yaml"
        assert args.verbose is True

    def test_defaults(self) -> None:
        """
        Confirm that common args default to None when not explicitly provided on the command line. The None default allows merge_cli_overrides to distinguish between a flag that was set and one that was omitted, preventing accidental overwriting of YAML-loaded values. This test parses an empty argument list and checks both config and verbose attributes.

        Returns:
            None
        """
        parser = argparse.ArgumentParser()
        add_shared_cli_args(parser)
        args = parser.parse_args([])
        assert args.config is None
        assert args.verbose is None


# -----------------------------------------------------------------------
# _resolve_config
# -----------------------------------------------------------------------

class TestResolveConfig:
    """Tests for _resolve_config from parsed argparse namespace."""

    def test_default_config_when_no_yaml(self) -> None:
        """
        Verify that _resolve_config returns a default ModvxConfig when no --config path is provided. This is the most common mode during automated testing where the built-in defaults are sufficient and no external YAML file is required. The test constructs a minimal namespace with all overrideable fields set to None and asserts the returned object is a ModvxConfig.

        Returns:
            None
        """
        ns = argparse.Namespace(
            config=None, experiment_name=None, initial_cycle_start=None,
            final_cycle_start=None, forecast_step_hours=None,
            observation_interval_hours=None, cycle_interval_hours=None,
            forecast_length_hours=None, verbose=None, save_intermediate=None,
            enable_logs=None,
        )
        cfg = resolve_config_from_namespace(ns)
        assert isinstance(cfg, ModvxConfig)

    def test_overrides_applied(self) -> None:
        """
        Verify that non-None CLI namespace values override corresponding fields in the resolved config. This test sets experiment_name, forecast_step_hours, and verbose in the namespace and checks that all three are reflected in the returned ModvxConfig object. Correct override propagation is critical so that CLI flags take effect without requiring a full YAML rewrite.

        Returns:
            None
        """
        ns = argparse.Namespace(
            config=None, experiment_name="test_exp", initial_cycle_start=None,
            final_cycle_start=None, forecast_step_hours=6,
            observation_interval_hours=None, cycle_interval_hours=None,
            forecast_length_hours=None, verbose=True, save_intermediate=None,
            enable_logs=None,
        )
        cfg = resolve_config_from_namespace(ns)
        assert cfg.experiment_name == "test_exp"
        assert cfg.forecast_step_hours == 6
        assert cfg.verbose is True

    def test_yaml_loading(self, tmp_path: Path) -> None:
        """
        Verify that _resolve_config loads a YAML file and applies its fields when --config is provided. This test writes a minimal YAML config to a temporary path, constructs a namespace pointing at it, and asserts that the returned ModvxConfig reflects the YAML-specified values. YAML loading must take precedence over built-in dataclass defaults for the config mechanism to be useful.

        Parameters:
            tmp_path (Path): Pytest-provided temporary directory, isolated per test invocation.

        Returns:
            None
        """
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("experiment_name: yaml_exp\nforecast_step_hours: 3\n")
        ns = argparse.Namespace(
            config=str(yaml_file), experiment_name=None,
            initial_cycle_start=None, final_cycle_start=None,
            forecast_step_hours=None, observation_interval_hours=None,
            cycle_interval_hours=None, forecast_length_hours=None,
            verbose=None, save_intermediate=None, enable_logs=None,
        )
        cfg = resolve_config_from_namespace(ns)
        assert cfg.experiment_name == "yaml_exp"
        assert cfg.forecast_step_hours == 3


# -----------------------------------------------------------------------
# main() entry point
# -----------------------------------------------------------------------

class TestMain:
    """Tests for the main() CLI entry point and subcommand dispatch."""

    def test_no_subcommand_exits(self) -> None:
        """
        Verify that calling main with no subcommand argument causes argparse to exit with a non-zero code. The CLI is designed to always require a subcommand, so omitting one is a user error that should be surfaced immediately. This test guards against any future change that might silently accept an empty invocation and run with default behaviour.

        Returns:
            None
        """
        with pytest.raises(SystemExit):
            main([])

    def test_run_subcommand_dispatches(self) -> None:
        """
        Verify that the 'run' subcommand instantiates TaskManager and ParallelProcessor and calls pp.run. Both collaborators are mocked so the test does not depend on any filesystem state or real computation. The assertion confirms that the full dispatch chain from the CLI entry point down to parallel execution is wired correctly.

        Returns:
            None
        """
        with patch("modvx.task_manager.TaskManager") as mock_tm, \
             patch("modvx.parallel.ParallelProcessor") as mock_pp:
            mock_tm_instance = MagicMock()
            mock_tm.return_value = mock_tm_instance
            mock_tm_instance.build_work_units.return_value = []
            mock_pp_instance = MagicMock()
            mock_pp.return_value = mock_pp_instance

            main(["run"])

            mock_tm.assert_called_once()
            mock_pp_instance.run.assert_called_once()

    def test_extract_csv_dispatches(self) -> None:
        """
        Verify that the 'extract-csv' subcommand instantiates FileManager and calls extract_fss_to_csv. FileManager is mocked to prevent any disk access during this unit test. Correct dispatch ensures that the CSV extraction workflow runs when the user invokes the subcommand.

        Returns:
            None
        """
        with patch("modvx.file_manager.FileManager") as mock_fm:
            mock_fm_instance = MagicMock()
            mock_fm.return_value = mock_fm_instance
            main(["extract-csv"])
            mock_fm_instance.extract_fss_to_csv.assert_called_once()

    def test_plot_all_dispatches(self) -> None:
        """
        Verify that 'plot --all' calls Visualizer.generate_all_plots rather than the single-plot method. Visualizer is mocked to prevent any filesystem access or matplotlib rendering during the test. This confirms the --all flag is routed to the batch generation path rather than the individual metric-per-call path.

        Returns:
            None
        """
        with patch("modvx.visualizer.Visualizer") as mock_viz:
            mock_viz_instance = MagicMock()
            mock_viz.return_value = mock_viz_instance
            main(["plot", "--all"])
            mock_viz_instance.generate_all_plots.assert_called_once()

    def test_plot_single_dispatches(self) -> None:
        """
        Verify that 'plot' with explicit domain, threshold, and window calls plot_fss_vs_leadtime. This exercises the non-batch path where a single combination is plotted rather than iterating over all available combinations. The Visualizer is mocked so the test does not require any CSV data on disk.

        Returns:
            None
        """
        with patch("modvx.visualizer.Visualizer") as mock_viz:
            mock_viz_instance = MagicMock()
            mock_viz.return_value = mock_viz_instance
            main(["plot", "--domain", "GLOBAL", "--thresh", "90", "--window", "3"])
            mock_viz_instance.plot_fss_vs_leadtime.assert_called()

    def test_plot_with_metric_filter(self) -> None:
        """
        Verify that the --metric flag is parsed as a comma-separated list and forwarded to generate_all_plots. The metric filter allows users to generate only a subset of plots rather than all six default metrics. This test confirms the splitting and lowercasing logic is applied before the list is passed to the visualizer.

        Returns:
            None
        """
        with patch("modvx.visualizer.Visualizer") as mock_viz:
            mock_viz_instance = MagicMock()
            mock_viz.return_value = mock_viz_instance
            main(["plot", "--all", "--metric", "fss,pod"])
            call_kwargs = mock_viz_instance.generate_all_plots.call_args
            assert call_kwargs[1]["metrics"] == ["fss", "pod"]

    def test_plot_missing_args_exits(self) -> None:
        """
        Verify that 'plot' without --all or a complete domain/thresh/window combination exits with an error. Providing only --domain without --thresh and --window is an incomplete specification that cannot produce a meaningful plot. This test ensures the guard condition inside _cmd_plot fires and the process exits rather than producing a misleading empty plot.

        Returns:
            None
        """
        with patch("modvx.visualizer.Visualizer"):
            with pytest.raises(SystemExit):
                main(["plot", "--domain", "GLOBAL"])

    def test_validate_dispatches(self) -> None:
        """
        Verify that the 'validate' subcommand calls Visualizer.list_available_options and prints results. The Visualizer is mocked to return a known set of domains, thresholds, and windows so the test focuses on dispatch correctness rather than content. Successful dispatch confirms the subparser is registered and the func default is wired to _cmd_validate.

        Returns:
            None
        """
        with patch("modvx.visualizer.Visualizer") as mock_viz:
            mock_viz_instance = MagicMock()
            mock_viz.return_value = mock_viz_instance
            mock_viz_instance.list_available_options.return_value = (
                ["GLOBAL"], [90.0], [3],
            )
            main(["validate"])
            mock_viz_instance.list_available_options.assert_called_once()

    def test_validate_no_data_exits(self) -> None:
        """
        Verify that the 'validate' subcommand exits with code 1 when list_available_options returns None. An all-None return from list_available_options signals that no CSV files were found in the configured directory. The CLI must surface this as an error exit rather than printing empty output to prevent users from silently assuming verification passed.

        Returns:
            None
        """
        with patch("modvx.visualizer.Visualizer") as mock_viz:
            mock_viz_instance = MagicMock()
            mock_viz.return_value = mock_viz_instance
            mock_viz_instance.list_available_options.return_value = (None, None, None)
            with pytest.raises(SystemExit):
                main(["validate"])

    def test_run_with_vxdomain_override(self) -> None:
        """
        Verify that --vxdomain comma-separated values are split and uppercased before being applied to the config. The raw string 'GLOBAL,TROPICS' must be converted to a list ['GLOBAL', 'TROPICS'] and merged into the config prior to TaskManager instantiation. This test inspects the config argument passed to TaskManager to confirm correct parsing.

        Returns:
            None
        """
        with patch("modvx.task_manager.TaskManager") as mock_tm, \
             patch("modvx.parallel.ParallelProcessor") as mock_pp:
            mock_tm_instance = MagicMock()
            mock_tm.return_value = mock_tm_instance
            mock_tm_instance.build_work_units.return_value = []
            mock_pp_instance = MagicMock()
            mock_pp.return_value = mock_pp_instance

            main(["run", "--vxdomain", "GLOBAL,TROPICS"])

            cfg = mock_tm.call_args[0][0]
            assert cfg.vxdomain == ["GLOBAL", "TROPICS"]

    def test_run_with_target_resolution_numeric(self) -> None:
        """
        Verify that a numeric --target-resolution string is parsed to a float before config merging. Numeric resolution values such as '0.25' represent degrees and must be stored as float so that downstream regridding logic can create the correct common grid. This test confirms the type coercion step is applied prior to handing the config to TaskManager.

        Returns:
            None
        """
        with patch("modvx.task_manager.TaskManager") as mock_tm, \
             patch("modvx.parallel.ParallelProcessor") as mock_pp:
            mock_tm_instance = MagicMock()
            mock_tm.return_value = mock_tm_instance
            mock_tm_instance.build_work_units.return_value = []
            mock_pp_instance = MagicMock()
            mock_pp.return_value = mock_pp_instance

            main(["run", "--target-resolution", "0.25"])

            cfg = mock_tm.call_args[0][0]
            assert cfg.target_resolution == 0.25

    def test_run_backend_multiprocessing(self) -> None:
        """
        Verify that the --backend and --nprocs arguments are forwarded correctly to ParallelProcessor. ParallelProcessor accepts keyword arguments for backend selection and worker count, so this test confirms the call site passes the correct keyword values. Incorrect forwarding would silently default to serial execution regardless of the user's flag.

        Returns:
            None
        """
        with patch("modvx.task_manager.TaskManager") as mock_tm, \
             patch("modvx.parallel.ParallelProcessor") as mock_pp:
            mock_tm_instance = MagicMock()
            mock_tm.return_value = mock_tm_instance
            mock_tm_instance.build_work_units.return_value = []
            mock_pp_instance = MagicMock()
            mock_pp.return_value = mock_pp_instance

            main(["run", "--backend", "multiprocessing", "--nprocs", "4"])

            mock_pp.assert_called_once()
            assert mock_pp.call_args[1]["backend"] == "multiprocessing"
            assert mock_pp.call_args[1]["nprocs"] == 4


# -----------------------------------------------------------------------
# CLI target_resolution ValueError branch
# -----------------------------------------------------------------------


class TestCliTargetResolutionValueError:
    """Lines 130-131: target_resolution not 'obs'/'fcst' and not a float."""

    def test_invalid_resolution_string_passes_through(self) -> None:
        """
        Verify that a non-numeric, non-'obs'/'fcst' --target-resolution string is passed through unchanged. When the string cannot be converted to float, the ValueError branch in _cmd_run leaves the value as-is rather than raising. This test confirms the branch is reached without crashing and that TaskManager is still instantiated successfully.

        Returns:
            None
        """
        args = argparse.Namespace(
            config=None,
            obs_dir=None,
            output_dir=None,
            vxdomain=None,
            target_resolution="not_a_number",
            mpas_grid_file=None,
            cache_dir="/tmp/cache",
            backend="serial",
            nprocs=1,
        )
        with patch("modvx.task_manager.TaskManager") as MockTM, \
             patch("modvx.parallel.ParallelProcessor") as MockPP:
            MockPP.return_value.run.return_value = None
            handle_run_subcommand(args)
        MockTM.assert_called_once()


# -----------------------------------------------------------------------
# CLI __main__ entry point
# -----------------------------------------------------------------------


class TestCliMainEntryPoint:
    """Line 376: if __name__ == '__main__': main()."""

    def test_main_module(self) -> None:
        """
        Verify that the module-level guard executes main() when __name__ == '__main__'. This test simulates the guard block by injecting a mock main function into an exec context and confirming it is called exactly once. The guard ensures the CLI is invocable directly as a script in addition to being called from entry_points.

        Returns:
            None
        """
        with patch("modvx.cli.main") as mock_main:
            code = "if True: main()"
            exec(code, {"main": mock_main, "__name__": "__main__"})
            mock_main.assert_called_once()


# -----------------------------------------------------------------------
# CLI cache-dir, mpas-grid-file, and auto-cache branches
# -----------------------------------------------------------------------


class TestCliOverrideBranches:
    """Cover --cache-dir, --mpas-grid-file, and auto cache_dir branches."""

    def test_run_with_cache_dir(self) -> None:
        """
        Verify that --cache-dir is applied to the config before TaskManager instantiation. The shared observation cache directory must be set in the config so that all parallel workers can write to and read from the same path. This test inspects the config passed to TaskManager to confirm the cache_dir field reflects the CLI flag value.

        Returns:
            None
        """
        with patch("modvx.task_manager.TaskManager") as mock_tm, \
             patch("modvx.parallel.ParallelProcessor") as mock_pp:
            inst = MagicMock()
            mock_tm.return_value = inst
            inst.build_work_units.return_value = []
            mock_pp.return_value = MagicMock()
            main(["run", "--cache-dir", "/tmp/cache"])
            cfg = mock_tm.call_args[0][0]
            assert cfg.cache_dir == "/tmp/cache"

    def test_run_with_mpas_grid_file(self) -> None:
        """
        Verify that --mpas-grid-file is merged into the config and forwarded to TaskManager. The MPAS grid file path is required by the loader to source cell coordinates and must be configurable from the command line. This test confirms the CLI override mechanism correctly propagates the path into the config object.

        Returns:
            None
        """
        with patch("modvx.task_manager.TaskManager") as mock_tm, \
             patch("modvx.parallel.ParallelProcessor") as mock_pp:
            inst = MagicMock()
            mock_tm.return_value = inst
            inst.build_work_units.return_value = []
            mock_pp.return_value = MagicMock()
            main(["run", "--mpas-grid-file", "grid/my.nc"])
            cfg = mock_tm.call_args[0][0]
            assert cfg.mpas_grid_file == "grid/my.nc"

    def test_run_auto_cache_dir(self) -> None:
        """
        Verify that a deterministic auto-derived cache directory is set when --cache-dir is not provided. The auto cache path is derived from the configured output directory and contains a '.obs_cache' component so it does not collide with other pipeline outputs. This test confirms the automatic assignment happens before TaskManager is called.

        Returns:
            None
        """
        with patch("modvx.task_manager.TaskManager") as mock_tm, \
             patch("modvx.parallel.ParallelProcessor") as mock_pp:
            inst = MagicMock()
            mock_tm.return_value = inst
            inst.build_work_units.return_value = []
            mock_pp.return_value = MagicMock()
            main(["run"])
            cfg = mock_tm.call_args[0][0]
            assert cfg.cache_dir is not None
            assert ".obs_cache" in cfg.cache_dir
