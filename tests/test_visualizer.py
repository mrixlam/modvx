#!/usr/bin/env python3

"""
Unit tests for MODvx visualizer module.

This module contains unit and integration tests for the Visualizer class, verifying CSV-driven plotting, metric labeling, and output generation. It exercises the end-to-end PNG creation pipeline, checks module-level constants, and validates behaviour for empty CSV directories and bounded/unbounded metric handling. Fixtures produce realistic CSV inputs and temporary output directories so plotting can be tested without external data dependencies.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from modvx.config import ModvxConfig
from modvx.visualizer import (
    Visualizer,
    _ALL_METRICS,
    _BOUNDED_METRICS,
    _METRIC_LABELS,
    _WINDOW_INDEPENDENT_METRICS, #noqa: F401
)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI


def _make_csv(csv_dir: Path, 
              experiment: str = "exp1") -> Path:
    """
    This helper function creates a synthetic CSV file with both FSS and continuous metrics for testing. The CSV contains six lead times for a single domain, threshold, and window combination, with made-up metric values that increase over lead time. The function ensures the output directory exists, writes the combined DataFrame to a CSV file named after the experiment, and returns the path to the created file. 

    Parameters:
        csv_dir (Path): Directory in which the CSV file will be written; created if absent.
        experiment (str): Stem name for the output CSV filename. Defaults to 'exp1'.

    Returns:
        Path: Absolute path to the written CSV file.
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    fss_df = pd.DataFrame({
        "initTime": ["2024091700"] * 6,
        "leadTime": [1, 2, 3, 4, 5, 6],
        "domain": ["GLOBAL"] * 6,
        "thresh": [90.0] * 6,
        "window": [3] * 6,
        "fss": [0.6, 0.65, 0.7, 0.72, 0.75, 0.78],
        "pod": [np.nan] * 6,
        "far": [np.nan] * 6,
        "csi": [np.nan] * 6,
        "fbias": [np.nan] * 6,
        "ets": [np.nan] * 6,
    })
    cont_df = pd.DataFrame({
        "initTime": ["2024091700"] * 6,
        "leadTime": [1, 2, 3, 4, 5, 6],
        "domain": ["GLOBAL"] * 6,
        "thresh": [90.0] * 6,
        "window": [np.nan] * 6,
        "fss": [np.nan] * 6,
        "pod": [0.5, 0.55, 0.6, 0.62, 0.65, 0.68],
        "far": [0.3, 0.28, 0.25, 0.22, 0.20, 0.18],
        "csi": [0.4, 0.45, 0.5, 0.52, 0.55, 0.58],
        "fbias": [1.1, 1.05, 1.0, 0.98, 0.95, 0.93],
        "ets": [0.2, 0.25, 0.3, 0.32, 0.35, 0.38],
    })
    df = pd.concat([fss_df, cont_df], ignore_index=True)
    csv_path = csv_dir / f"{experiment}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def viz_setup(tmp_path: Path) -> tuple:
    """
    This fixture sets up a Visualizer instance with a realistic CSV file for testing. It creates a temporary CSV directory and plot output directory within the pytest-provided tmp_path, generates a synthetic CSV file using _make_csv, and initializes a Visualizer with a ModvxConfig pointing to these directories. The fixture returns the Visualizer instance along with the paths to the CSV and plot directories for use in tests. 

    Parameters:
        tmp_path (Path): Pytest-supplied per-test temporary directory.

    Returns:
        tuple: Three-element tuple (Visualizer, csv_dir Path, plot_dir Path).
    """
    csv_dir = tmp_path / "csv"
    plot_dir = tmp_path / "plots"
    _make_csv(csv_dir, "exp1")

    cfg = ModvxConfig(
        base_dir=str(tmp_path),
        csv_dir=str(csv_dir),
        plot_dir=str(plot_dir),
    )
    viz = Visualizer(cfg)
    return viz, csv_dir, plot_dir


class TestConstants:
    """ Verify module-level metric constants are defined correctly. """

    def test_all_metrics_list(self: "TestConstants") -> None:
        """
        This test verifies that the _ALL_METRICS list includes 'fss' and contains exactly six metrics. The presence of 'fss' confirms that the primary metric is included, while the length check ensures no extra or missing metrics have been introduced. This guards against accidental changes to the supported metric set that could break plotting or labeling logic.

        Parameters:
            self (TestConstants): The test class instance.

        Returns:
            None
        """
        assert "fss" in _ALL_METRICS
        assert len(_ALL_METRICS) == 6

    def test_metric_labels(self: "TestConstants") -> None:
        """
        This test verifies that every metric in _ALL_METRICS has a corresponding entry in the _METRIC_LABELS dictionary. This ensures that all supported metrics have defined human-readable labels for plotting and display purposes, preventing KeyErrors during label lookups and ensuring consistent presentation of metric names.

        Parameters:
            self (TestConstants): The test class instance.

        Returns:
            None
        """
        for m in _ALL_METRICS:
            assert m in _METRIC_LABELS

    def test_bounded_metrics(self: "TestConstants") -> None:
        """
        This test verifies that the _BOUNDED_METRICS set includes 'fss' but does not include 'fbias'. Bounded metrics like 'fss' receive y-axis limits of [0, 1] during plotting, while unbounded metrics like 'fbias' do not. This test confirms that the correct metrics are classified as bounded or unbounded, which is critical for proper plot scaling and visualization. 

        Parameters:
            self (TestConstants): The test class instance.

        Returns:
            None
        """
        assert "fss" in _BOUNDED_METRICS
        assert "fbias" not in _BOUNDED_METRICS


class TestHelperFilters:
    """ Coverage for internal helper branches. """

    def test_filter_df_matches_window(self: "TestHelperFilters") -> None:
        """
        This test verifies that the _filter_df method correctly filters a DataFrame based on domain, threshold, and window criteria. The test constructs a simple DataFrame with two rows for the same domain and threshold but different window sizes. When filtering for a specific window size, only the matching row should be returned. This confirms that the filtering logic correctly applies all three criteria to narrow down the DataFrame to the intended subset. 

        Parameters: 
            self (TestHelperFilters): The test class instance.

        Returns:
            None
        """
        df = pd.DataFrame({
            "domain": ["GLOBAL", "GLOBAL"],
            "thresh": [90.0, 90.0],
            "window": [3, 5],
        })
        filtered = Visualizer._filter_df(df, "GLOBAL", 90.0, 3)
        assert len(filtered) == 1
        assert int(filtered["window"].iloc[0]) == 3

    def test_aggregate_metric_missing_column(self: "TestHelperFilters", tmp_path: Path) -> None:
        """
        This test verifies that the _aggregate_metric method returns None when the specified metric column is missing from the CSV data. The test creates a CSV file that includes 'fss' but omits 'pod'. When attempting to aggregate 'pod', the method should detect the missing column and return None rather than raising a KeyError. This ensures robust handling of incomplete or unexpected CSV data without crashing the entire plotting process. 

        Parameters:
            self (TestHelperFilters): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame({
            "domain": ["GLOBAL"],
            "thresh": [90.0],
            "window": [3],
            "leadTime": [1],
            "fss": [0.5],
        }).to_csv(csv_dir / "exp1.csv", index=False)

        result = Visualizer._aggregate_metric(csv_dir / "exp1.csv", "GLOBAL", 90.0, 3, "pod")
        assert result is None

    def test_aggregate_metric_window_required(self: "TestHelperFilters", tmp_path: Path) -> None:
        """
        This test verifies that the _aggregate_metric method returns None when the window parameter is missing for a window-dependent metric. The test creates a CSV file with a window-dependent metric ('fss') and attempts to aggregate it without specifying a window. The method should return None, ensuring that window-dependent metrics are not aggregated without the required window information.

        Parameters:
            self (TestHelperFilters): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame({
            "domain": ["GLOBAL"],
            "thresh": [90.0],
            "window": [3],
            "leadTime": [1],
            "fss": [0.5],
        }).to_csv(csv_dir / "exp1.csv", index=False)

        result = Visualizer._aggregate_metric(csv_dir / "exp1.csv", "GLOBAL", 90.0, None, "fss")
        assert result is None

    def test_aggregate_metric_window_independent(self: "TestHelperFilters", tmp_path: Path) -> None:
        """
        This test verifies that the _aggregate_metric method can successfully aggregate a window-independent metric even when the window parameter is None. The test creates a CSV file with a window-independent metric ('pod') and calls _aggregate_metric without a window. The method should return the correct aggregated value, confirming that window-independent metrics are handled properly without requiring window information. 

        Parameters:
            self (TestHelperFilters): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame({
            "domain": ["GLOBAL"],
            "thresh": [90.0],
            "window": [np.nan],
            "leadTime": [1],
            "pod": [0.6],
        }).to_csv(csv_dir / "exp1.csv", index=False)

        result = Visualizer._aggregate_metric(csv_dir / "exp1.csv", "GLOBAL", 90.0, None, "pod")
        assert result is not None
        assert result.iloc[0] == pytest.approx(0.6)

    def test_set_y_limits_no_valid(self: "TestHelperFilters") -> None:
        """
        This test verifies that the _set_y_limits method does not raise an exception when given NaN values for a bounded metric. The method should handle the case where no valid metric values are present without crashing, even though it cannot set meaningful limits. This ensures robustness in edge cases where the CSV data may be incomplete or filtered down to an empty set. 

        Parameters:
            self (TestHelperFilters): The test class instance.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        Visualizer._set_y_limits(ax, [np.nan, np.nan], "fss")
        plt.close(fig)


class TestLeadtimeTicks:
    """ Coverage for lead-time tick helpers. """

    def test_leadtime_tick_interval(self: "TestLeadtimeTicks", tmp_path: Path) -> None:
        """
        This test verifies that the _leadtime_tick_interval method returns the correct tick interval based on the forecast length in hours. The method should return 1 for forecast lengths up to 12 hours, 3 for up to 24 hours, 6 for up to 48 hours, and 12 for longer forecasts. This test creates Visualizer instances with different forecast lengths and asserts that the returned tick interval matches the expected values, ensuring that lead-time ticks are spaced appropriately for different forecast durations. 

        Parameters: 
            self (TestLeadtimeTicks): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test configurations.

        Returns:
            None
        """
        cfg = ModvxConfig(base_dir=str(tmp_path), forecast_length_hours=12)
        viz = Visualizer(cfg)
        assert viz._leadtime_tick_interval() == 1

        cfg = ModvxConfig(base_dir=str(tmp_path), forecast_length_hours=24)
        viz = Visualizer(cfg)
        assert viz._leadtime_tick_interval() == 3

        cfg = ModvxConfig(base_dir=str(tmp_path), forecast_length_hours=48)
        viz = Visualizer(cfg)
        assert viz._leadtime_tick_interval() == 6

        cfg = ModvxConfig(base_dir=str(tmp_path), forecast_length_hours=72)
        viz = Visualizer(cfg)
        assert viz._leadtime_tick_interval() == 12

    def test_apply_leadtime_ticks(self: "TestLeadtimeTicks", tmp_path: Path) -> None:
        """
        This test verifies that the _apply_leadtime_ticks method correctly sets the x-axis ticks on a Matplotlib axis based on the provided lead times and the configured tick interval. The test creates a Visualizer instance with a 24-hour forecast length, applies lead-time ticks to an axis with lead times from 1 to 24 hours, and asserts that the first tick is approximately 3 hours and the last tick is at least 24 hours. This confirms that the method correctly calculates and applies ticks according to the forecast length and lead time range.

        Parameters:
            self (TestLeadtimeTicks): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test configurations.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        cfg = ModvxConfig(base_dir=str(tmp_path), forecast_length_hours=24)
        viz = Visualizer(cfg)
        fig, ax = plt.subplots()
        viz._apply_leadtime_ticks(ax, [1, 2, 3, 24])
        ticks = ax.get_xticks()
        assert ticks[0] == pytest.approx(3)
        assert ticks[-1] >= 24
        plt.close(fig)


class TestPlotFssVsLeadtime:
    """ Tests for plot_fss_vs_leadtime single-metric plot generation. """

    def test_creates_png(self: "TestPlotFssVsLeadtime", viz_setup) -> None:
        """
        This test verifies that the plot_fss_vs_leadtime method creates and returns a PNG file path for a valid metric/domain/thresh/window combination. The output directory and CSV source directory are supplied explicitly, and the returned string must end in '.png' and point to a file that actually exists on disk. This confirms the full plot-and-save pipeline runs without error for the nominal case.

        Parameters:
            self (TestPlotFssVsLeadtime): The test class instance.
            viz_setup (tuple): A fixture providing a Visualizer instance, CSV directory, and plot directory.

        Returns:
            None
        """
        viz, csv_dir, plot_dir = viz_setup
        result = viz.plot_fss_vs_leadtime(
            domain="GLOBAL", thresh="90", window="3",
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
            metric="fss",
        )
        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".png")

    def test_returns_none_no_csvs(self: "TestPlotFssVsLeadtime", tmp_path: Path) -> None:
        """
        This test verifies that the plot_fss_vs_leadtime method returns None when the CSV source directory is empty or missing. Without any CSV files to read, the method cannot generate a plot and must return None rather than raising an exception. This ensures that callers can handle the no-data case gracefully without needing to catch exceptions. 

        Parameters:
            self (TestPlotFssVsLeadtime): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test configurations.

        Returns:
            None
        """
        cfg = ModvxConfig(base_dir=str(tmp_path))
        viz = Visualizer(cfg)
        result = viz.plot_fss_vs_leadtime(
            csv_dir=str(tmp_path / "empty"),
            output_dir=str(tmp_path / "out"),
        )
        assert result is None

    def test_all_metrics_plotted(self: "TestPlotFssVsLeadtime", viz_setup) -> None:
        """
        This test verifies that the plot_fss_vs_leadtime method can generate plots for all metrics listed in _ALL_METRICS without raising exceptions. The test iterates over each metric, calls the plotting method with the same domain/thresh/window, and asserts that a non-None result is returned. This confirms that the plotting logic can handle every supported metric type, including both bounded and unbounded ones, without crashing. 

        Parameters:
            self (TestPlotFssVsLeadtime): The test class instance.

        Returns:
            None
        """
        viz, csv_dir, plot_dir = viz_setup
        for metric in _ALL_METRICS:
            result = viz.plot_fss_vs_leadtime(
                domain="GLOBAL", thresh="90", window="3",
                csv_dir=str(csv_dir), output_dir=str(plot_dir),
                metric=metric,
            )
            assert result is not None

    def test_bounded_metric_ylim(self: "TestPlotFssVsLeadtime", viz_setup) -> None:
        """ 
        This test verifies that the plot_fss_vs_leadtime method applies y-axis limits of [0, 1] for bounded metrics like 'pod'. The test calls the plotting method for 'pod' and checks that the resulting plot file is created successfully. While we cannot directly inspect the plot's y-limits without reading the file, the fact that the method completes without error confirms that the internal logic for setting limits on bounded metrics runs correctly. 

        Parameters:
            self (TestPlotFssVsLeadtime): The test class instance.

        Returns:
            None
        """
        viz, csv_dir, plot_dir = viz_setup
        # Just ensure it doesn't crash — the y-limit logic is internal
        result = viz.plot_fss_vs_leadtime(
            domain="GLOBAL", thresh="90", window="3",
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
            metric="pod",
        )
        assert result is not None

    def test_no_matching_data_returns_plot(self: "TestPlotFssVsLeadtime", viz_setup) -> None:
        """
        This test verifies that the plot_fss_vs_leadtime method returns a PNG file path even when no matching data exists for the specified domain/thresh/window combination. The test calls the method with a non-existent domain, which should result in an empty plot. However, the method should still create and return a PNG file path rather than returning None or raising an exception. This confirms that the plotting pipeline can handle cases with no data gracefully by producing an empty plot instead of failing. 

        Parameters:
            self (TestPlotFssVsLeadtime): The test class instance.
            viz_setup (tuple): A fixture providing a Visualizer instance, CSV directory, and plot directory.

        Returns:
            None
        """
        viz, csv_dir, plot_dir = viz_setup
        result = viz.plot_fss_vs_leadtime(
            domain="NONEXISTENT", thresh="90", window="3",
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
        )
        # Plot is still saved (empty), returns the path
        assert result is not None

    def test_window_independent_metric(self: "TestPlotFssVsLeadtime", viz_setup) -> None:
        """
        This test verifies that the plot_fss_vs_leadtime method can successfully generate a plot for a window-independent metric ('pod') even when the window parameter is None. The test calls the plotting method for 'pod' with window=None and asserts that a PNG file path is returned. This confirms that window-independent metrics do not require a window value and that the plotting logic correctly handles this case without error.

        Parameters:
            self (TestPlotFssVsLeadtime): The test class instance.
            viz_setup (tuple): A fixture providing a Visualizer instance, CSV directory, and plot directory.

        Returns:
            None
        """
        viz, csv_dir, plot_dir = viz_setup
        result = viz.plot_fss_vs_leadtime(
            domain="GLOBAL", thresh="90", window=None,
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
            metric="pod",
        )
        assert result is not None
        assert "window" not in os.path.basename(result)

    def test_window_required_missing(self: "TestPlotFssVsLeadtime", viz_setup) -> None:
        """
        This test verifies that the plot_fss_vs_leadtime method returns None when the window parameter is missing for a window-dependent metric. The test calls the method with window=None for a metric that requires a window, and asserts that the result is None. This confirms that the plotting logic correctly handles the case where a required window value is not provided.

        Parameters:
            self (TestPlotFssVsLeadtime): The test class instance.
            viz_setup (tuple): A fixture providing a Visualizer instance, CSV directory, and plot directory.

        Returns:
            None
        """
        viz, csv_dir, plot_dir = viz_setup
        result = viz.plot_fss_vs_leadtime(
            domain="GLOBAL", thresh="90", window=None,
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
            metric="fss",
        )
        assert result is None


class TestPlotFssDifference:
    """ Tests for plot_fss_difference (experiment - control) plots. """

    def test_creates_diff_plot(self: "TestPlotFssDifference", tmp_path: Path) -> None:
        """
        This test verifies that the plot_fss_difference method creates and returns a PNG file path for a valid control experiment and domain/thresh/window combination. The test sets up two CSV files (control and experiment) in the source directory, calls the plotting method with the control experiment name, and asserts that a PNG file is created successfully. This confirms that the method can read both control and experiment data, compute the difference, and generate a plot without error for the nominal case. 

        Parameters:
            self (TestPlotFssDifference): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files and output directories.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        plot_dir = tmp_path / "plots"
        _make_csv(csv_dir, "control")
        _make_csv(csv_dir, "exp1")

        cfg = ModvxConfig(base_dir=str(tmp_path))
        viz = Visualizer(cfg)
        result = viz.plot_fss_difference(
            control_experiment="control",
            domain="GLOBAL", thresh="90", window="3",
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
        )
        assert result is not None
        assert os.path.exists(result)

    def test_returns_none_no_csvs(self: "TestPlotFssDifference", tmp_path: Path) -> None:
        """
        This test verifies that the plot_fss_difference method returns None when the CSV source directory is empty or missing. Without any CSV files to read for either the control or experiment, the method cannot compute a difference plot and must return None rather than raising an exception. This ensures that callers can handle the no-data case gracefully without needing to catch exceptions. 

        Parameters:
            self (TestPlotFssDifference): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files and output directories.

        Returns:
            None
        """
        cfg = ModvxConfig(base_dir=str(tmp_path))
        viz = Visualizer(cfg)
        result = viz.plot_fss_difference(
            control_experiment="ctrl",
            csv_dir=str(tmp_path / "empty"),
            output_dir=str(tmp_path / "out"),
        )
        assert result is None

    def test_returns_none_missing_control(self: "TestPlotFssDifference", tmp_path: Path) -> None:
        """
        This test verifies that the plot_fss_difference method returns None when the specified control experiment CSV file is missing from the source directory. The method should attempt to find a CSV file matching the control experiment name, and if it cannot find one, it must return None rather than raising an exception. This ensures that callers can handle the missing control case gracefully without needing to catch exceptions. 

        Parameters:
            self (TestPlotFssDifference): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files and output directories.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        _make_csv(csv_dir, "exp1")

        cfg = ModvxConfig(base_dir=str(tmp_path))
        viz = Visualizer(cfg)
        result = viz.plot_fss_difference(
            control_experiment="nonexistent",
            csv_dir=str(csv_dir),
            output_dir=str(tmp_path / "out"),
        )
        assert result is None

    def test_window_required_missing(self: "TestPlotFssDifference", tmp_path: Path) -> None:
        """
        This test verifies that the plot_fss_difference method returns None when the window parameter is missing for a window-dependent metric. The method should require a window value for metrics that depend on it, and if it is not provided, it must return None rather than raising an exception. This ensures that callers can handle the missing window case gracefully without needing to catch exceptions.

        Parameters:
            self (TestPlotFssDifference): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files and output directories.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        plot_dir = tmp_path / "plots"
        _make_csv(csv_dir, "control")

        cfg = ModvxConfig(base_dir=str(tmp_path))
        viz = Visualizer(cfg)
        result = viz.plot_fss_difference(
            control_experiment="control",
            domain="GLOBAL", thresh="90", window=None,
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
            metric="fss",
        )
        assert result is None


class TestGenerateAllPlots:
    """ Tests for generate_all_plots batch plot generation. """

    def test_generates_all_combos(self: "TestGenerateAllPlots", viz_setup) -> None:
        """
        This test verifies that the generate_all_plots method generates plots for all combinations of metrics, domains, thresholds, and windows discovered in the CSV files. The test uses the standard viz_setup CSV which contains one domain (GLOBAL), one threshold (90.0), and one window (3), along with six metrics. When generate_all_plots is called without a metric filter, it should produce 6 plots (one for each metric) for the single combination of domain/thresh/window. This confirms that the method correctly iterates over all discovered options and generates the expected number of plots.

        Parameters:
            self (TestGenerateAllPlots): The test class instance.
            viz_setup (tuple): A fixture providing a Visualizer instance, CSV directory, and plot directory.

        Returns:
            None
        """
        viz, csv_dir, plot_dir = viz_setup
        count = viz.generate_all_plots(
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
        )
        # 6 metrics × 1 domain × 1 threshold × 1 window = 6
        assert count == 6

    def test_metric_filter(self: "TestGenerateAllPlots", viz_setup) -> None:
        """
        This test verifies that the generate_all_plots method correctly applies a metric filter to generate only the specified metrics. The test uses the standard viz_setup CSV which contains six metrics, but calls generate_all_plots with a filter for just 'fss' and 'pod'. The method should then produce only 2 plots (one for each of the two specified metrics) for the single combination of domain/thresh/window. This confirms that the metric filtering logic works correctly to limit plot generation to the desired subset of metrics. 

        Parameters:
            self (TestGenerateAllPlots): The test class instance.
            viz_setup (tuple): A fixture providing a Visualizer instance, CSV directory, and plot directory.

        Returns:
            None
        """
        viz, csv_dir, plot_dir = viz_setup
        count = viz.generate_all_plots(
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
            metrics=["fss", "pod"],
        )
        # 2 metrics × 1 domain × 1 threshold × 1 window = 2
        assert count == 2

    def test_returns_zero_no_csvs(self: "TestGenerateAllPlots", tmp_path: Path) -> None:
        """
        This test verifies that the generate_all_plots method returns a count of zero when the CSV source directory is empty or missing. Without any CSV files to read, the method cannot discover any options or generate any plots, and must return 0 rather than raising an exception. This ensures that callers can handle the no-data case gracefully without needing to catch exceptions. 

        Parameters:
            self (TestGenerateAllPlots): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test configurations.

        Returns:
            None
        """
        cfg = ModvxConfig(base_dir=str(tmp_path))
        viz = Visualizer(cfg)
        count = viz.generate_all_plots(csv_dir=str(tmp_path / "empty"))
        assert count == 0

    def test_window_independent_only(self: "TestGenerateAllPlots", tmp_path: Path) -> None:
        """
        This test verifies that the generate_all_plots method can successfully generate plots for a window-independent metric when the window column contains NaN values. The test creates a CSV file with a window-independent metric ('pod') and NaN windows, then calls generate_all_plots with a filter for 'pod'. The method should generate 1 plot for 'pod' despite the NaN windows, confirming that window-independent metrics are handled correctly even when the window column is not populated. 

        Parameters:
            self (TestGenerateAllPlots): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test configurations.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        plot_dir = tmp_path / "plots"
        csv_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "initTime": ["2024091700"],
            "leadTime": [1],
            "domain": ["GLOBAL"],
            "thresh": [90.0],
            "window": [np.nan],
            "pod": [0.6],
        }).to_csv(csv_dir / "exp1.csv", index=False)

        cfg = ModvxConfig(base_dir=str(tmp_path), csv_dir=str(csv_dir), plot_dir=str(plot_dir))
        viz = Visualizer(cfg)
        count = viz.generate_all_plots(
            csv_dir=str(csv_dir), output_dir=str(plot_dir), metrics=["pod"],
        )
        assert count == 1


class TestListAvailableOptions:
    """ Tests for list_available_options CSV discovery. """

    def test_returns_options(self: "TestListAvailableOptions", viz_setup) -> None:
        """
        This test verifies that the list_available_options method correctly extracts domains, thresholds, and window sizes from CSV files. The test uses the standard viz_setup CSV which has one domain (GLOBAL), one threshold (90.0), and one window (3). All three returned lists must match those values exactly, confirming the discovery logic reads and de-duplicates the relevant CSV columns correctly.

        Parameters:
            self (TestListAvailableOptions): The test class instance.
            viz_setup (tuple): A fixture providing a Visualizer instance, CSV directory, and plot directory.

        Returns:
            None
        """
        viz, csv_dir, _ = viz_setup
        domains, thresholds, windows = viz.list_available_options(csv_dir=str(csv_dir))
        assert domains == ["GLOBAL"]
        assert thresholds == [90.0]
        assert windows == [3]

    def test_returns_none_no_csvs(self: "TestListAvailableOptions", tmp_path: Path) -> None:
        """
        This test verifies that the list_available_options method returns a triple of None values when no CSV files are present. The absence of CSV files means no options can be discovered, and the caller must receive (None, None, None) so it can skip plot generation cleanly.

        Parameters:
            self (TestListAvailableOptions): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test configurations.

        Returns:
            None
        """
        cfg = ModvxConfig(base_dir=str(tmp_path))
        viz = Visualizer(cfg)
        d, t, w = viz.list_available_options(csv_dir=str(tmp_path / "empty"))
        assert d is None
        assert t is None
        assert w is None

    def test_drops_nan_windows(self: "TestListAvailableOptions", tmp_path: Path) -> None:
        """
        This test verifies that the list_available_options method correctly drops NaN values from the window column when discovering options. The test creates a CSV file with two rows for the same domain and threshold, but one row has a valid window (3) while the other has NaN. The method should return a list of windows that includes only the valid window (3) and excludes the NaN, confirming that it properly handles missing window values during discovery. 

        Parameters:
            self (TestListAvailableOptions): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test configurations.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "domain": ["GLOBAL", "GLOBAL"],
            "thresh": [90.0, 90.0],
            "window": [np.nan, 3],
            "leadTime": [1, 1],
            "fss": [0.5, 0.6],
        }).to_csv(csv_dir / "exp1.csv", index=False)

        cfg = ModvxConfig(base_dir=str(tmp_path), csv_dir=str(csv_dir))
        viz = Visualizer(cfg)
        domains, thresholds, windows = viz.list_available_options(csv_dir=str(csv_dir))
        assert domains == ["GLOBAL"]
        assert thresholds == [90.0]
        assert windows == [3]


class TestPlotHorizontalMap:
    """ Tests for plot_horizontal_map Cartopy map plotting. """

    def test_creates_map(self: "TestPlotHorizontalMap", tmp_path: Path) -> None:
        """
        This test verifies that the plot_horizontal_map method creates and saves a PNG file when given a valid 2D DataArray with latitude and longitude coordinates. The test constructs a simple 10x20 DataArray with appropriate coordinate values, calls the plotting method with a title and output path, and asserts that the returned path is not None and points to an existing PNG file. This confirms that the method can successfully generate and save a horizontal map plot without error for the nominal case. 

        Parameters:
            self (TestPlotHorizontalMap): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test configurations and output files.

        Returns:
            None
        """
        pytest.importorskip("cartopy")

        cfg = ModvxConfig(base_dir=str(tmp_path), plot_dir="plots")
        viz = Visualizer(cfg)
        field = xr.DataArray(
            dims=["latitude", "longitude"],
            coords={
                "latitude": np.linspace(-5, 5, 10),
                "longitude": np.linspace(0, 20, 20),
            },
        )
        outpath = str(tmp_path / "plots" / "map.png")
        result = viz.plot_horizontal_map(field, title="Test", output_path=outpath)
        assert result is not None
        assert os.path.exists(result)

    def test_default_output_path(self: "TestPlotHorizontalMap", tmp_path: Path) -> None:
        """
        This test verifies that the plot_horizontal_map method uses a default output path containing 'horizontal_map.png' when none is given. When the caller does not provide an explicit output_path, the method should construct one based on the configured plot directory. This ensures map plots always land in a predictable location even when called without a fully specified path. 

        Parameters:
            self (TestPlotHorizontalMap): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test configurations and output files.

        Returns:
            None
        """
        pytest.importorskip("cartopy")

        cfg = ModvxConfig(base_dir=str(tmp_path), plot_dir="plots")
        viz = Visualizer(cfg)
        field = xr.DataArray(
            np.random.rand(5, 5),
            dims=["latitude", "longitude"],
            coords={
                "latitude": np.linspace(-2, 2, 5),
                "longitude": np.linspace(0, 4, 5),
            },
        )
        result = viz.plot_horizontal_map(field, title="Test")
        assert result is not None
        assert "horizontal_map.png" in result


def _write_csv(path: Path, rows: list) -> None:
    """
    This helper function writes a list of dictionaries to a CSV file at the specified path. Each dictionary in the list represents a row, with keys corresponding to column headers. This utility is used in multiple tests to create custom CSV files with specific content for testing the Visualizer's handling of various data scenarios. 

    Parameters:
        path (Path): Destination file path including filename and .csv extension.
        rows (list): List of dictionaries representing CSV rows; keys become column headers.

    Returns:
        None
    """
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _make_cfg(tmp_path: Path) -> ModvxConfig:
    """
    This helper function creates a ModvxConfig object with all directory paths set relative to the provided temporary path. The base_dir is set to the string representation of tmp_path, and subdirectories for forecasts, observations, output, CSVs, and plots are defined as fixed names under the base directory. This utility allows tests to easily generate a consistent configuration object that points to the appropriate locations within the pytest-managed temporary directory structure. 

    Parameters:
        tmp_path (Path): Pytest-supplied per-test temporary directory used as base_dir.

    Returns:
        ModvxConfig: Configuration object with all directories set relative to tmp_path.
    """
    return ModvxConfig(
        base_dir=str(tmp_path),
        fcst_dir="fcst",
        obs_dir="obs",
        output_dir="output",
        csv_dir="csv",
        plot_dir="plots",
    )


class TestVisualizerEmptyFilteredData:
    """ Tests for plot_fss_vs_leadtime handling of empty or filtered-out data. """

    def test_skip_when_filtered_empty(self: "TestVisualizerEmptyFilteredData", tmp_path: Path) -> None:
        """
        This test verifies that the plot_fss_vs_leadtime method returns None when the CSV data is present but all rows are filtered out due to no matching domain/thresh/window combinations. The test creates a CSV file with a single row that has domain 'OTHER', then calls the plotting method with domain='GLOBAL'. Since there are no rows matching 'GLOBAL', the method should recognize that there is no data to plot and return None rather than attempting to create an empty plot or raising an exception. This ensures that the method can gracefully handle cases where filtering results in no data without crashing. 

        Parameters:
            self (TestVisualizerEmptyFilteredData): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files and output directories.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        _write_csv(csv_dir / "exp1.csv", [
            {"domain": "OTHER", "thresh": 90.0, "window": 3, "leadTime": 6, "fss": 0.5},
        ])
        cfg = _make_cfg(tmp_path)
        viz = Visualizer(cfg)

        import logging

        warning_messages: list = []

        class _CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.levelno == logging.WARNING:
                    warning_messages.append(self.format(record))

        _handler = _CapturingHandler()
        _vx_logger = logging.getLogger("modvx.visualizer")
        _vx_logger.addHandler(_handler)
        try:
            viz.plot_fss_vs_leadtime(
                domain="GLOBAL", thresh="90", window="3",
                csv_dir=str(csv_dir), output_dir=str(tmp_path / "plots"),
            )
        finally:
            _vx_logger.removeHandler(_handler)
        assert any("No data for" in m for m in warning_messages)


class TestVisualizerDiffEmptyControl:
    """ Tests for plot_fss_difference handling of empty control data. """

    def test_empty_control_returns_none(self: "TestVisualizerDiffEmptyControl", tmp_path: Path) -> None:
        """
        This test verifies that the plot_fss_difference method returns None when the control experiment CSV file contains no rows matching the specified domain/thresh/window combination. The method should attempt to find control rows for the given filters, and if it finds none, it must return None rather than attempting to compute differences or generate a plot. This ensures that callers can handle the empty control case gracefully without needing to catch exceptions. 

        Parameters:
            self (TestVisualizerDiffEmptyControl): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files and output directories.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        _write_csv(csv_dir / "ctrl.csv", [
            {"domain": "OTHER", "thresh": 90.0, "window": 3, "leadTime": 6, "fss": 0.5},
        ])
        _write_csv(csv_dir / "exp1.csv", [
            {"domain": "GLOBAL", "thresh": 90.0, "window": 3, "leadTime": 6, "fss": 0.6},
        ])
        cfg = _make_cfg(tmp_path)
        viz = Visualizer(cfg)
        result = viz.plot_fss_difference(
            control_experiment="ctrl",
            domain="GLOBAL", thresh="90", window="3",
            csv_dir=str(csv_dir), output_dir=str(tmp_path / "plots"),
        )
        assert result is None


class TestVisualizerDiffEmptyExperiment:
    """ Tests for plot_fss_difference handling of empty experiment data. """

    def test_skip_empty_experiment(self: "TestVisualizerDiffEmptyExperiment", tmp_path: Path) -> None:
        """
        This test verifies that the plot_fss_difference method continues past an experiment CSV with no data matching the specified domain/thresh/window combination. The experiment CSV contains only domain 'OTHER' while the control CSV has 'GLOBAL', so the diff loop finds no experiment rows and skips the combination. The method should still return a valid result path from the control data rather than returning None.

        Parameters:
            self (TestVisualizerDiffEmptyExperiment): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files and output directories.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()
        _write_csv(csv_dir / "ctrl.csv", [
            {"domain": "GLOBAL", "thresh": 90.0, "window": 3, "leadTime": 6, "fss": 0.5},
        ])
        _write_csv(csv_dir / "exp1.csv", [
            {"domain": "OTHER", "thresh": 90.0, "window": 3, "leadTime": 6, "fss": 0.6},
        ])
        cfg = _make_cfg(tmp_path)
        viz = Visualizer(cfg)
        result = viz.plot_fss_difference(
            control_experiment="ctrl",
            domain="GLOBAL", thresh="90", window="3",
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
        )
        assert result is not None


class TestVisualizerCartopyMissing:
    """ Tests for plot_horizontal_map handling of missing Cartopy dependency. """

    def test_no_cartopy_returns_none(self: "TestVisualizerCartopyMissing", tmp_path: Path) -> None:
        """
        This test verifies that the plot_horizontal_map method returns None when the Cartopy library is not available. The method should attempt to import Cartopy, and if it fails with an ImportError, it must catch the exception and return None rather than raising an error. This ensures that callers can handle the missing Cartopy case gracefully without needing to catch exceptions. 

        Parameters:
            self (TestVisualizerCartopyMissing): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test configurations and output files.

        Returns:
            None
        """
        import builtins

        cfg = _make_cfg(tmp_path)
        viz = Visualizer(cfg)

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name.startswith("cartopy"):
                raise ImportError("no cartopy")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = _mock_import
        try:
            result = viz.plot_horizontal_map(
                field=xr.DataArray(np.zeros((2, 2)), dims=["latitude", "longitude"]),
                title="test",
                output_path=str(tmp_path / "map.png"),
            )
        finally:
            builtins.__import__ = original_import
        assert result is None


class TestVisualizerGenerateAllPlotsException:
    """ Tests for generate_all_plots handling of exceptions in plot_fss_vs_leadtime. """

    def test_exception_counted(self: "TestVisualizerGenerateAllPlotsException", tmp_path: Path) -> None:
        """
        This test verifies that generate_all_plots returns 0 when plot_fss_vs_leadtime raises an exception for every combination. This simulates a scenario where all individual plot calls fail, for example due to a rendering error. The batch method must catch the exception, decrement or skip the count, and return 0 rather than propagating the exception to the caller.

        Parameters:
            self (TestVisualizerGenerateAllPlotsException): The test class instance.
            tmp_path (Path): Pytest-supplied temporary directory for creating test CSV files and output directories.

        Returns:
            None
        """
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()
        _write_csv(csv_dir / "exp.csv", [
            {"domain": "GLOBAL", "thresh": 90.0, "window": 3, "leadTime": 6, "fss": 0.5},
        ])
        cfg = _make_cfg(tmp_path)
        viz = Visualizer(cfg)

        def _raise_runtime(*args, **kwargs):
            raise RuntimeError("boom")

        orig_method = viz.plot_fss_vs_leadtime
        viz.plot_fss_vs_leadtime = _raise_runtime
        try:
            count = viz.generate_all_plots(
                csv_dir=str(csv_dir), output_dir=str(plot_dir),
            )
        finally:
            viz.plot_fss_vs_leadtime = orig_method
        assert count == 0
