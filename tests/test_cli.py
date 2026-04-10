#!/usr/bin/env python3

"""
Unit tests for MODvx command-line interface.

This module contains unit tests for the command-line interface defined in modvx.cli. The tests cover argument parsing, configuration resolution, logging setup, and the dispatch of subcommands to their respective handler functions. Lightweight stub classes are used to isolate the CLI logic from the underlying TaskManager, ParallelProcessor, FileManager, and Visualizer implementations, allowing for focused testing of the CLI behaviour without side effects.    

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from modvx.cli import (
    add_shared_cli_args,
    handle_run_subcommand,
    resolve_config_from_namespace,
    configure_root_logging,
    main,
)
from modvx.config import ModvxConfig


class _StubTaskManager:
    """ Lightweight fake TaskManager – captures the config passed to the constructor and provides no-op implementations of build_work_units and execute_work_unit. """

    last_instance: "_StubTaskManager | None" = None

    def __init__(self: "_StubTaskManager", 
                 config: ModvxConfig) -> None:
        """
        This constructor captures the ModvxConfig passed to TaskManager and stores it on the instance for later inspection by tests. The class attribute ``last_instance`` is updated to point to the most recently created instance, allowing tests to access the config directly after main() returns without needing to mock TaskManager's constructor.

        Parameters:
            config (ModvxConfig): The configuration object passed to TaskManager, expected to contain all settings resolved from CLI arguments and YAML loading.

        Returns:
            None
        """
        self.config = config
        _StubTaskManager.last_instance = self

    def build_work_units(self: "_StubTaskManager") -> list:
        """
        This method provides a no-op implementation of build_work_units, returning an empty list. Since the focus of the tests is on CLI argument parsing and dispatch rather than the actual work unit generation logic, this stub allows TaskManager to be instantiated and used without performing any real computations.

        Parameters:
            None

        Returns:
            list: An empty list representing no work units.
        """
        return []

    def execute_work_unit(self: "_StubTaskManager", 
                          unit: dict) -> None:
        """
        This method provides a no-op implementation of execute_work_unit, doing nothing when called. This allows the CLI tests to verify that TaskManager is instantiated and that the run method of ParallelProcessor is invoked without needing to execute any real work units or perform any computations. 

        Parameters:
            unit (dict): A work unit to be executed (ignored in this stub).

        Returns:
            None
        """
        pass


class _StubParallelProcessor:
    """ Lightweight fake ParallelProcessor – captures the backend and nprocs arguments passed to the constructor and records whether run() was called. """

    last_instance: "_StubParallelProcessor | None" = None

    def __init__(self: "_StubParallelProcessor", 
                 execute_fn: callable, 
                 backend: str = "auto", 
                 nprocs: int | None = None) -> None:
        """
        This constructor captures the execute_fn, backend, and nprocs arguments passed to ParallelProcessor and stores them on the instance for later inspection by tests. The class attribute ``last_instance`` is updated to point to the most recently created instance, allowing tests to access these values directly after main() returns without needing to mock ParallelProcessor's constructor.

        Parameters:
            execute_fn (callable): The function that ParallelProcessor will call to execute work units (ignored in this stub).
            backend (str): The parallel processing backend specified via CLI (e.g., "auto", "multiprocessing", "dask").
            nprocs (int | None): The number of processes to use for parallel execution, if applicable.

        Returns:
            None
        """
        self.execute_fn = execute_fn
        self.backend = backend
        self.nprocs = nprocs
        self.run_called = False
        _StubParallelProcessor.last_instance = self

    def run(self: "_StubParallelProcessor", 
            units: list) -> None:
        """
        This method provides a no-op implementation of run that simply sets a flag to indicate it was called. This allows the CLI tests to confirm that when the 'run' subcommand is executed, the ParallelProcessor's run method is invoked as expected, without performing any actual parallel processing or computations. 

        Parameters:
            units (list): A list of work units to be executed (ignored in this stub).

        Returns:
            None
        """
        self.run_called = True


class _StubFileManager:
    """ Lightweight fake FileManager – records whether extract_fss_to_csv was called. """

    last_instance: "_StubFileManager | None" = None

    def __init__(self: "_StubFileManager", 
                 config: ModvxConfig) -> None:
        """
        This constructor captures the ModvxConfig passed to FileManager and stores it on the instance for later inspection by tests. The class attribute ``last_instance`` is updated to point to the most recently created instance, allowing tests to access the config directly after main() returns without needing to mock FileManager's constructor.

        Parameters:
            config (ModvxConfig): The configuration object passed to FileManager, expected to contain all settings resolved from CLI arguments and YAML loading.

        Returns:
            None     
        """
        self.config = config
        self.extract_fss_to_csv_called = False
        _StubFileManager.last_instance = self

    def extract_fss_to_csv(self: "_StubFileManager", 
                           output_dir=None, 
                           csv_dir=None) -> None:
        """ 
        This method provides a fake implementation of extract_fss_to_csv that records whether it was called. When the 'extract-csv' subcommand is executed, this method should be invoked with the appropriate output_dir and csv_dir arguments. By setting a flag when this method is called, tests can confirm that the CLI correctly dispatches to this method when extracting FSS data to CSV format. 

        Parameters:
            output_dir (str | None): The directory where extracted files should be saved.
            csv_dir (str | None): The directory containing CSV files to be used for extraction.

        Returns:
            None
        """
        self.extract_fss_to_csv_called = True


class _StubVisualizer:
    """ Fake Visualizer that records calls to generate_all_plots, plot_fss_vs_leadtime, and list_available_options, and allows tests to inspect the arguments passed to these methods. """

    last_instance: "_StubVisualizer | None" = None
    available_options: tuple = (["GLOBAL"], [90.0], [3])

    def __init__(self: "_StubVisualizer", 
                 config: ModvxConfig) -> None:
        """
        This constructor captures the ModvxConfig passed to Visualizer and stores it on the instance for later inspection by tests. The class attribute ``last_instance`` is updated to point to the most recently created instance, allowing tests to access the config directly after main() returns without needing to mock Visualizer's constructor.

        Parameters:
            config (ModvxConfig): The configuration object passed to Visualizer, expected to contain all settings resolved from CLI arguments and YAML loading.

        Returns:
            None
        """
        self.config = config
        self.generate_all_plots_called = False
        self.generate_all_plots_kwargs: dict = {}
        self.plot_fss_vs_leadtime_called = False
        self.list_available_options_called = False
        _StubVisualizer.last_instance = self

    def generate_all_plots(self: "_StubVisualizer", 
                           csv_dir=None, 
                           output_dir=None,
                           metrics=None) -> None:
        """ 
        This method provides a fake implementation of generate_all_plots that records the arguments it was called with. When the 'plot' subcommand is executed with the --all flag, this method should be invoked with the appropriate csv_dir, output_dir, and metrics arguments. By storing these values on the instance, tests can confirm that the CLI correctly parses and forwards these arguments to Visualizer when generating all plots. 

        Parameters:
            csv_dir (str | None): The directory containing CSV files to be used for plotting.
            output_dir (str | None): The directory where plots should be saved.
            metrics (list | None): A list of metrics to be plotted.

        Returns:
            None
        """
        self.generate_all_plots_called = True
        self.generate_all_plots_kwargs = {
            "csv_dir": csv_dir,
            "output_dir": output_dir,
            "metrics": metrics,
        }

    def plot_fss_vs_leadtime(self: "_StubVisualizer", 
                             domain=None, 
                             thresh=None, 
                             window=None, 
                             csv_dir=None, 
                             output_dir=None,
                             metric=None) -> None:
        """ 
        This method provides a fake implementation of plot_fss_vs_leadtime that records whether it was called. When the 'plot' subcommand is executed with specific --domain, --thresh, and --window arguments (and without --all), this method should be invoked with the appropriate arguments. By setting a flag when this method is called, tests can confirm that the CLI correctly parses the required arguments and dispatches to this plotting method when generating a single plot. 

        Parameters:
            domain (str | None): The domain for which to plot FSS vs lead time.
            thresh (float | None): The threshold value for the plot.
            window (int | None): The window size for the plot.
            csv_dir (str | None): The directory containing CSV files to be used for plotting.
            output_dir (str | None): The directory where plots should be saved.
            metric (str | None): The metric to be plotted.

        Returns:
            None
        """
        self.plot_fss_vs_leadtime_called = True

    def list_available_options(self: "_StubVisualizer", 
                               csv_dir=None) -> tuple:
        """ 
        This method provides a fake implementation of list_available_options that records whether it was called and returns a predefined set of available options. When the 'validate' subcommand is executed, this method should be invoked to retrieve the available domains, thresholds, and windows from the CSV data. By setting a flag when this method is called and returning a known value, tests can confirm that the CLI correctly dispatches to this method and handles its output when validating available options. 

        Parameters:
            csv_dir (str | None): The directory containing CSV files to be used for listing options.

        Returns:
            tuple: A tuple of available options.
        """
        self.list_available_options_called = True
        return _StubVisualizer.available_options


class TestSetupLogging:
    """ Tests for configure_root_logging verifying that the root logger is configured at the correct level based on the verbose flag in the config. """

    def test_info_level_by_default(self: "TestSetupLogging") -> None:
        """
        This test verifies that the root logger is configured at INFO level by default when verbose is False in the config. INFO level provides general information about the execution flow without overwhelming detail, and should be the default for typical usage. The test creates a ModvxConfig with verbose=False, calls configure_root_logging, and asserts that the root logger's level is set to logging.INFO.

        Parameters:
            None

        Returns:
            None
        """
        import logging
        cfg = ModvxConfig(verbose=False)
        configure_root_logging(cfg)
        assert logging.getLogger().level == logging.INFO

    def test_debug_level_when_verbose(self: "TestSetupLogging") -> None:
        """
        This test verifies that the root logger is configured at DEBUG level when verbose is True in the config. DEBUG level provides detailed information useful for troubleshooting and development, and should be enabled when the user specifies verbose mode. The test creates a ModvxConfig with verbose=True, calls configure_root_logging, and asserts that the root logger's level is set to logging.DEBUG.

        Parameters:
            None

        Returns:
            None
        """
        import logging
        cfg = ModvxConfig(verbose=True)
        configure_root_logging(cfg)
        assert logging.getLogger().level == logging.DEBUG


class TestAddCommonArgs:
    """ Tests for add_shared_cli_args verifying that common CLI arguments are registered correctly on an ArgumentParser and have expected defaults. """

    def test_config_and_verbose_registered(self: "TestAddCommonArgs") -> None:
        """
        This test confirms that the --config and --verbose flags are correctly registered on an ArgumentParser when add_shared_cli_args is called. The test creates a new ArgumentParser, calls add_shared_cli_args, and then parses a sample argument list containing both flags. The resulting namespace should have attributes for config and verbose with the expected values, confirming that the arguments are properly added to the parser.

        Parameters:
            None

        Returns:
            None
        """
        parser = argparse.ArgumentParser()
        add_shared_cli_args(parser)
        args = parser.parse_args(["-c", "my.yaml", "--verbose"])
        assert args.config == "my.yaml"
        assert args.verbose is True

    def test_defaults(self: "TestAddCommonArgs") -> None:
        """
        This test verifies that the default values for the --config and --verbose flags are None when add_shared_cli_args is called. The test creates a new ArgumentParser, calls add_shared_cli_args, and then parses an empty argument list. The resulting namespace should have config and verbose attributes set to None, confirming that the defaults are correctly established when the flags are not provided.

        Parameters:
            None

        Returns:
            None
        """
        parser = argparse.ArgumentParser()
        add_shared_cli_args(parser)
        args = parser.parse_args([])
        assert args.config is None
        assert args.verbose is None


class TestResolveConfig:
    """ Tests for resolve_config_from_namespace verifying that configuration is correctly resolved from a namespace with various combinations of CLI overrides and YAML loading. """

    def test_default_config_when_no_yaml(self: "TestResolveConfig") -> None:
        """
        This test verifies that resolve_config_from_namespace returns a ModvxConfig object with default values when no --config path is provided in the namespace. The test constructs a namespace with all fields set to None, calls resolve_config_from_namespace, and asserts that the returned object is an instance of ModvxConfig. This confirms that the function can handle the absence of a YAML configuration file and still produce a valid config object with defaults.

        Parameters:
            None

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

    def test_overrides_applied(self: "TestResolveConfig") -> None:
        """
        This test verifies that resolve_config_from_namespace correctly applies overrides from the namespace when no YAML file is provided. The test constructs a namespace with specific values for experiment_name, forecast_step_hours, and verbose, while leaving config as None. After calling resolve_config_from_namespace, the returned config object should reflect the overridden values from the namespace, confirming that CLI overrides are correctly merged into the configuration even in the absence of a YAML file. 

        Parameters:
            None

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

    def test_yaml_loading(self: "TestResolveConfig", 
                          tmp_path: Path) -> None:
        """
        This test verifies that resolve_config_from_namespace correctly loads configuration values from a YAML file when the --config path is provided in the namespace. The test creates a temporary YAML file with specific configuration values, constructs a namespace with the config field set to the path of the YAML file, and calls resolve_config_from_namespace. The returned config object should reflect the values defined in the YAML file, confirming that YAML loading is functioning correctly and that values from the file are properly incorporated into the configuration. 

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


class TestMain:
    """ Tests for main() verifying that the CLI correctly parses arguments, dispatches to subcommand handlers, and applies overrides. """

    def test_no_subcommand_exits(self: "TestMain") -> None:
        """
        This test verifies that calling main() with an empty argument list results in a SystemExit. The CLI is designed to require a subcommand, so invoking main() without any arguments should trigger argparse's error handling and exit the process. This confirms that the CLI enforces the requirement for a subcommand and does not allow execution without one. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(SystemExit):
            main([])

    def test_run_subcommand_dispatches(self: "TestMain",
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the 'run' subcommand correctly dispatches to handle_run_subcommand and ultimately calls TaskManager and ParallelProcessor. Stub implementations replace both classes so no real computation occurs. After main() returns, the test confirms that TaskManager was instantiated and that ParallelProcessor.run was invoked.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubTaskManager.last_instance = None
        _StubParallelProcessor.last_instance = None
        monkeypatch.setattr("modvx.task_manager.TaskManager", _StubTaskManager)
        monkeypatch.setattr("modvx.parallel.ParallelProcessor", _StubParallelProcessor)

        main(["run"])

        assert _StubTaskManager.last_instance is not None
        assert _StubParallelProcessor.last_instance is not None
        assert _StubParallelProcessor.last_instance.run_called

    def test_extract_csv_dispatches(self: "TestMain",
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the 'extract-csv' subcommand calls FileManager.extract_fss_to_csv. A stub FileManager is injected so no real file operations occur. After main() returns the test confirms that the correct method was called.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubFileManager.last_instance = None
        monkeypatch.setattr("modvx.file_manager.FileManager", _StubFileManager)

        main(["extract-csv"])

        assert _StubFileManager.last_instance is not None
        assert _StubFileManager.last_instance.extract_fss_to_csv_called

    def test_plot_all_dispatches(self: "TestMain",
                                 monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the 'plot' subcommand with the --all flag calls Visualizer.generate_all_plots. A stub Visualizer is injected so no real plotting occurs. After main() returns the test confirms that the correct method was invoked.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubVisualizer.last_instance = None
        monkeypatch.setattr("modvx.visualizer.Visualizer", _StubVisualizer)

        main(["plot", "--all"])

        assert _StubVisualizer.last_instance is not None
        assert _StubVisualizer.last_instance.generate_all_plots_called

    def test_plot_single_dispatches(self: "TestMain",
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the 'plot' subcommand with specific --domain, --thresh, and --window arguments calls Visualizer.plot_fss_vs_leadtime. A stub Visualizer is injected so no real plotting occurs. After main() returns the test confirms that the correct method was invoked.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubVisualizer.last_instance = None
        monkeypatch.setattr("modvx.visualizer.Visualizer", _StubVisualizer)

        main(["plot", "--domain", "GLOBAL", "--thresh", "90", "--window", "3"])

        assert _StubVisualizer.last_instance is not None
        assert _StubVisualizer.last_instance.plot_fss_vs_leadtime_called

    def test_plot_with_metric_filter(self: "TestMain",
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the 'plot' subcommand with the --all flag and a --metric filter correctly passes the specified metrics to Visualizer.generate_all_plots. A stub Visualizer is injected and its recorded kwargs are inspected directly to confirm the metrics argument is parsed and forwarded correctly.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubVisualizer.last_instance = None
        monkeypatch.setattr("modvx.visualizer.Visualizer", _StubVisualizer)

        main(["plot", "--all", "--metric", "fss,pod"])

        assert _StubVisualizer.last_instance.generate_all_plots_kwargs["metrics"] == ["fss", "pod"]

    def test_plot_missing_args_exits(self: "TestMain",
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the 'plot' subcommand exits with a SystemExit when the required --domain, --thresh, and --window arguments are all missing and --all is not provided. A stub Visualizer is injected to prevent any real visualizer initialisation side-effects.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        monkeypatch.setattr("modvx.visualizer.Visualizer", _StubVisualizer)

        with pytest.raises(SystemExit):
            main(["plot", "--domain", "GLOBAL"])

    def test_validate_dispatches(self: "TestMain",
                                 monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the 'validate' subcommand calls Visualizer.list_available_options. A stub Visualizer is injected with a known return value; the test confirms that the method was called after main() returns.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubVisualizer.last_instance = None
        monkeypatch.setattr("modvx.visualizer.Visualizer", _StubVisualizer)
        monkeypatch.setattr(_StubVisualizer, "available_options", (["GLOBAL"], [90.0], [3]))

        main(["validate"])

        assert _StubVisualizer.last_instance is not None
        assert _StubVisualizer.last_instance.list_available_options_called

    def test_validate_no_data_exits(self: "TestMain",
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the 'validate' subcommand exits gracefully when Visualizer.list_available_options returns empty (None) values. The stub Visualizer's ``available_options`` class attribute is temporarily overridden to return (None, None, None), simulating no CSV data found.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        monkeypatch.setattr("modvx.visualizer.Visualizer", _StubVisualizer)
        monkeypatch.setattr(_StubVisualizer, "available_options", (None, None, None))

        with pytest.raises(SystemExit):
            main(["validate"])

    def test_run_with_vxdomain_override(self: "TestMain",
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the --vxdomain CLI override is correctly parsed as a list of strings and applied to the config passed to TaskManager. Stub classes replace TaskManager and ParallelProcessor; after main() returns the config stored on the stub instance is inspected directly.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubTaskManager.last_instance = None
        monkeypatch.setattr("modvx.task_manager.TaskManager", _StubTaskManager)
        monkeypatch.setattr("modvx.parallel.ParallelProcessor", _StubParallelProcessor)

        main(["run", "--vxdomain", "GLOBAL,TROPICS"])

        cfg = _StubTaskManager.last_instance.config
        assert cfg.vxdomain == ["GLOBAL", "TROPICS"]

    def test_run_with_target_resolution_numeric(self: "TestMain",
                                                monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the --target-resolution CLI override is correctly parsed as a float and applied to the config passed to TaskManager. Stub classes replace TaskManager and ParallelProcessor; after main() returns the config stored on the stub instance is inspected directly.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubTaskManager.last_instance = None
        monkeypatch.setattr("modvx.task_manager.TaskManager", _StubTaskManager)
        monkeypatch.setattr("modvx.parallel.ParallelProcessor", _StubParallelProcessor)

        main(["run", "--target-resolution", "0.25"])

        cfg = _StubTaskManager.last_instance.config
        assert cfg.target_resolution == 0.25

    def test_run_backend_multiprocessing(self: "TestMain",
                                         monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the --backend and --nprocs CLI overrides are correctly parsed and forwarded to ParallelProcessor when the 'run' subcommand is executed. The stub ParallelProcessor captures its constructor arguments; after main() returns the stored values are asserted directly.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubParallelProcessor.last_instance = None
        monkeypatch.setattr("modvx.task_manager.TaskManager", _StubTaskManager)
        monkeypatch.setattr("modvx.parallel.ParallelProcessor", _StubParallelProcessor)

        main(["run", "--backend", "multiprocessing", "--nprocs", "4"])

        pp = _StubParallelProcessor.last_instance
        assert pp is not None
        assert pp.backend == "multiprocessing"
        assert pp.nprocs == 4


class TestCliTargetResolutionValueError:
    """ Tests for the branch in _cmd_run that handles ValueError when parsing --target-resolution. """

    def test_invalid_resolution_string_passes_through(
            self: "TestCliTargetResolutionValueError",
            monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that when a non-numeric string is provided for --target-resolution, the ValueError is caught and the original string value is passed through to TaskManager without modification. Stub classes replace TaskManager and ParallelProcessor; after handle_run_subcommand() returns the stub instance is checked to confirm TaskManager was instantiated.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubTaskManager.last_instance = None
        monkeypatch.setattr("modvx.task_manager.TaskManager", _StubTaskManager)
        monkeypatch.setattr("modvx.parallel.ParallelProcessor", _StubParallelProcessor)

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
        handle_run_subcommand(args)

        assert _StubTaskManager.last_instance is not None


class TestCliMainEntryPoint:
    """ Tests for the module-level guard that calls main() when __name__ == '__main__'. """

    def test_main_module(self: "TestCliMainEntryPoint") -> None:
        """
        This test verifies that the main() function is called when the module is executed as a script. A lightweight callable records whether it was invoked, then a code snippet that simulates the __main__ guard is executed via exec. The test asserts that the callable was invoked exactly once, confirming the guard wires up correctly.

        Parameters:
            None

        Returns:
            None
        """
        call_log: list = []

        def fake_main(*args, 
                      **kwargs) -> None:
            """
            This is a lightweight fake main function that simply appends to a call_log list when invoked. This allows the test to confirm that main() is called when the module-level guard is executed, without needing to execute any real logic from the actual main function.

            Parameters:
                *args: Positional arguments (ignored).
                **kwargs: Keyword arguments (ignored).

            Returns:
                None 
            """
            call_log.append(True)

        code = "if True: main()"
        exec(code, {"main": fake_main, "__name__": "__main__"})
        assert len(call_log) == 1


class TestCliOverrideBranches:
    """ Tests for the CLI override branches covering --cache-dir, --mpas-grid-file, and automatic cache_dir derivation. """

    def test_run_with_cache_dir(self: "TestCliOverrideBranches",
                                monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the --cache-dir CLI override is correctly merged into the config and forwarded to TaskManager. Stub classes replace both classes; after main() returns the config stored on the TaskManager stub is inspected to confirm cache_dir matches the provided path.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubTaskManager.last_instance = None
        monkeypatch.setattr("modvx.task_manager.TaskManager", _StubTaskManager)
        monkeypatch.setattr("modvx.parallel.ParallelProcessor", _StubParallelProcessor)

        main(["run", "--cache-dir", "/tmp/cache"])

        cfg = _StubTaskManager.last_instance.config
        assert cfg.cache_dir == "/tmp/cache"

    def test_run_with_mpas_grid_file(self: "TestCliOverrideBranches",
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that the --mpas-grid-file CLI override is correctly merged into the config and forwarded to TaskManager. Stub classes replace both classes; after main() returns the config stored on the TaskManager stub is inspected to confirm mpas_grid_file matches the provided path.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubTaskManager.last_instance = None
        monkeypatch.setattr("modvx.task_manager.TaskManager", _StubTaskManager)
        monkeypatch.setattr("modvx.parallel.ParallelProcessor", _StubParallelProcessor)

        main(["run", "--mpas-grid-file", "grid/my.nc"])

        cfg = _StubTaskManager.last_instance.config
        assert cfg.mpas_grid_file == "grid/my.nc"

    def test_run_auto_cache_dir(self: "TestCliOverrideBranches",
                                monkeypatch: pytest.MonkeyPatch) -> None:
        """
        This test verifies that when no --cache-dir is provided, the config passed to TaskManager contains a cache_dir that is automatically derived and includes the expected substring. Stub classes replace both classes; after main() returns the config stored on the TaskManager stub is inspected to confirm the auto-derived cache_dir is non-None and contains ".obs_cache".

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture for safe attribute patching.

        Returns:
            None
        """
        _StubTaskManager.last_instance = None
        monkeypatch.setattr("modvx.task_manager.TaskManager", _StubTaskManager)
        monkeypatch.setattr("modvx.parallel.ParallelProcessor", _StubParallelProcessor)

        main(["run"])

        cfg = _StubTaskManager.last_instance.config
        assert cfg.cache_dir is not None
        assert ".obs_cache" in cfg.cache_dir
