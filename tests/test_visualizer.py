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
from unittest.mock import patch

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


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _make_csv(csv_dir: Path, experiment: str = "exp1") -> Path:
    """
    Create a minimal CSV file mimicking the output of extract_fss_to_csv for use in plot tests. The file contains six lead-time rows with realistic FSS, POD, FAR, CSI, FBIAS, and ETS values for a single GLOBAL domain entry at 90th-percentile threshold and window size 3. This shared helper avoids repeating DataFrame construction boilerplate across every visualizer test fixture.

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
    Create a Visualizer backed by real CSV data and pre-created output directories for plot tests. The fixture writes one CSV file via _make_csv, builds a ModvxConfig pointing at the temporary directories, and constructs a Visualizer instance. The returned tuple gives tests direct access to the Visualizer, the CSV directory, and the plot output directory without re-creating them.

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


# -----------------------------------------------------------------------
# Module-level constants
# -----------------------------------------------------------------------

class TestConstants:
    """Verify module-level metric constants are defined correctly."""

    def test_all_metrics_list(self) -> None:
        """
        Verify that the _ALL_METRICS constant contains 'fss' and lists exactly six metric keys. The six canonical metrics are fss, pod, far, csi, fbias, and ets. Having a fixed count guards against accidental deletion or duplication when new metrics are added to the module.

        Returns:
            None
        """
        assert "fss" in _ALL_METRICS
        assert len(_ALL_METRICS) == 6

    def test_metric_labels(self) -> None:
        """
        Verify that every metric listed in _ALL_METRICS has a corresponding entry in _METRIC_LABELS. Plot axis titles are generated from this mapping, so a missing label would silently fall back to the raw key string. This test iterates over all known metrics and confirms each one is present as a key in the label dictionary.

        Returns:
            None
        """
        for m in _ALL_METRICS:
            assert m in _METRIC_LABELS

    def test_bounded_metrics(self) -> None:
        """
        Verify that _BOUNDED_METRICS includes 'fss' but not 'fbias'. Bounded metrics receive clamped y-axis limits of [0, 1] during plotting, while unbounded metrics such as FBIAS can exceed 1 and should not be constrained. This test confirms the expected membership for the two representative metrics.

        Returns:
            None
        """
        assert "fss" in _BOUNDED_METRICS
        assert "fbias" not in _BOUNDED_METRICS


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

class TestHelperFilters:
    """Coverage for internal helper branches."""

    def test_filter_df_matches_window(self) -> None:
        """
        Verify that _filter_df returns only rows matching domain, threshold, and window.

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

    def test_aggregate_metric_missing_column(self, tmp_path: Path) -> None:
        """
        Verify that _aggregate_metric returns None when the requested metric column is missing.

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

    def test_aggregate_metric_window_required(self, tmp_path: Path) -> None:
        """
        Verify that _aggregate_metric returns None when window is missing for a window-dependent metric.

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

    def test_aggregate_metric_window_independent(self, tmp_path: Path) -> None:
        """
        Verify that _aggregate_metric ignores window for threshold-only metrics.

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

    def test_set_y_limits_no_valid(self) -> None:
        """
        Verify that _set_y_limits returns without error when no finite values exist.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        Visualizer._set_y_limits(ax, [np.nan, np.nan], "fss")
        plt.close(fig)


class TestLeadtimeTicks:
    """Coverage for lead-time tick helpers."""

    def test_leadtime_tick_interval(self, tmp_path: Path) -> None:
        """
        Verify that lead-time tick intervals follow forecast length rules.

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

    def test_apply_leadtime_ticks(self, tmp_path: Path) -> None:
        """
        Verify that _apply_leadtime_ticks applies ticks using the configured interval.

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


# -----------------------------------------------------------------------
# plot_fss_vs_leadtime
# -----------------------------------------------------------------------

class TestPlotFssVsLeadtime:
    """Tests for plot_fss_vs_leadtime single-metric plot generation."""

    def test_creates_png(self, viz_setup) -> None:
        """
        Verify that plot_fss_vs_leadtime creates and returns a PNG file path for a valid metric/domain/thresh/window combination. The output directory and CSV source directory are supplied explicitly, and the returned string must end in '.png' and point to a file that actually exists on disk. This confirms the full plot-and-save pipeline runs without error for the nominal case.

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

    def test_returns_none_no_csvs(self, tmp_path: Path) -> None:
        """
        Verify that plot_fss_vs_leadtime returns None when the CSV directory is empty or missing. Without any CSV data the method has nothing to plot and should return None to indicate no output was generated, allowing callers to skip downstream file processing for missing data.

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

    def test_all_metrics_plotted(self, viz_setup) -> None:
        """
        Verify that plot_fss_vs_leadtime can generate plots for all six supported metrics without error. Each metric in _ALL_METRICS is plotted independently for the same domain, threshold, and window combination, and the returned path must be non-None confirming a PNG was saved for every metric. This guards against missing label mappings or unsupported metric names causing silent failures.

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

    def test_bounded_metric_ylim(self, viz_setup) -> None:
        """
        Verify that plotting a bounded metric such as 'pod' does not raise an exception. Bounded metrics receive y-axis limits of [0, 1] during plot generation; this test confirms the y-limit clamping logic does not interfere with normal plot creation and that the result path is still returned.

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

    def test_no_matching_data_returns_plot(self, viz_setup) -> None:
        """
        Verify that plot_fss_vs_leadtime still writes a plot file even when the domain filter matches no rows. When the filtered DataFrame is empty after domain filtering, the method logs a warning and saves an empty plot rather than returning None. This ensures the caller always receives a valid path and does not need to handle the filtered-empty case as a special error condition.

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

    def test_window_independent_metric(self, viz_setup) -> None:
        """
        Verify that plot_fss_vs_leadtime works with window-independent metrics when window is None.

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

    def test_window_required_missing(self, viz_setup) -> None:
        """
        Verify that plot_fss_vs_leadtime returns None when window is missing for a window-dependent metric.

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


# -----------------------------------------------------------------------
# plot_fss_difference
# -----------------------------------------------------------------------

class TestPlotFssDifference:
    """Tests for plot_fss_difference (experiment - control) plots."""

    def test_creates_diff_plot(self, tmp_path: Path) -> None:
        """
        Verify that plot_fss_difference creates a PNG when both control and experiment CSV files exist. The method reads the control and experiment CSVs, computes lead-time differences, and saves a difference plot. This test provides both files and asserts the returned path exists on disk, confirming the full difference-plot pipeline runs correctly.

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

    def test_returns_none_no_csvs(self, tmp_path: Path) -> None:
        """
        Verify that plot_fss_difference returns None when the CSV source directory is empty or missing. Without any CSV files there is no control or experiment data to compare, so the method must return None rather than raising an exception.

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

    def test_returns_none_missing_control(self, tmp_path: Path) -> None:
        """
        Verify that plot_fss_difference returns None when the named control CSV does not exist. If the control experiment file is missing the comparison cannot be performed, so None must be returned instead of raising a FileNotFoundError or KeyError.

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

    def test_window_required_missing(self, tmp_path: Path) -> None:
        """
        Verify that plot_fss_difference returns None when window is missing for a window-dependent metric.

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


# -----------------------------------------------------------------------
# generate_all_plots
# -----------------------------------------------------------------------

class TestGenerateAllPlots:
    """Tests for generate_all_plots batch plot generation."""

    def test_generates_all_combos(self, viz_setup) -> None:
        """
        Verify that generate_all_plots creates a separate PNG for every metric-domain-threshold-window combination. With one domain, one threshold, one window, and six metrics from the CSV, the expected count is 6. This integration test confirms the batch plotting loop iterates correctly over all discovered option combinations without skipping or duplicating any.

        Returns:
            None
        """
        viz, csv_dir, plot_dir = viz_setup
        count = viz.generate_all_plots(
            csv_dir=str(csv_dir), output_dir=str(plot_dir),
        )
        # 6 metrics × 1 domain × 1 threshold × 1 window = 6
        assert count == 6

    def test_metric_filter(self, viz_setup) -> None:
        """
        Verify that the 'metrics' argument filters generate_all_plots to only the requested metric subset. When metrics=['fss', 'pod'] is supplied, only those two are plotted per combination, yielding 2 plots instead of the full 6. This confirms the filter is applied before the loop rather than generating all plots and discarding the unwanted ones.

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

    def test_returns_zero_no_csvs(self, tmp_path: Path) -> None:
        """
        Verify that generate_all_plots returns 0 when no CSV files exist in the source directory. When list_available_options finds no CSVs it returns None for all option lists, and generate_all_plots should return 0 plots generated rather than raising an exception.

        Returns:
            None
        """
        cfg = ModvxConfig(base_dir=str(tmp_path))
        viz = Visualizer(cfg)
        count = viz.generate_all_plots(csv_dir=str(tmp_path / "empty"))
        assert count == 0

    def test_window_independent_only(self, tmp_path: Path) -> None:
        """
        Verify that generate_all_plots handles window-independent metrics without requiring window values.

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


# -----------------------------------------------------------------------
# list_available_options
# -----------------------------------------------------------------------

class TestListAvailableOptions:
    """Tests for list_available_options CSV discovery."""

    def test_returns_options(self, viz_setup) -> None:
        """
        Verify that list_available_options correctly extracts domains, thresholds, and window sizes from CSV files. The test uses the standard viz_setup CSV which has one domain (GLOBAL), one threshold (90.0), and one window (3). All three returned lists must match those values exactly, confirming the discovery logic reads and de-duplicates the relevant CSV columns correctly.

        Returns:
            None
        """
        viz, csv_dir, _ = viz_setup
        domains, thresholds, windows = viz.list_available_options(csv_dir=str(csv_dir))
        assert domains == ["GLOBAL"]
        assert thresholds == [90.0]
        assert windows == [3]

    def test_returns_none_no_csvs(self, tmp_path: Path) -> None:
        """
        Verify that list_available_options returns a triple of None values when no CSV files are present. The absence of CSV files means no options can be discovered, and the caller must receive (None, None, None) so it can skip plot generation cleanly.

        Returns:
            None
        """
        cfg = ModvxConfig(base_dir=str(tmp_path))
        viz = Visualizer(cfg)
        d, t, w = viz.list_available_options(csv_dir=str(tmp_path / "empty"))
        assert d is None
        assert t is None
        assert w is None

    def test_drops_nan_windows(self, tmp_path: Path) -> None:
        """
        Verify that list_available_options drops NaN window values.

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


# -----------------------------------------------------------------------
# plot_horizontal_map
# -----------------------------------------------------------------------

class TestPlotHorizontalMap:
    """Tests for plot_horizontal_map Cartopy map plotting."""

    def test_creates_map(self, tmp_path: Path) -> None:
        """
        Verify that plot_horizontal_map creates a PNG file at the specified output path when Cartopy is available. The test skips gracefully if Cartopy is not installed using pytest.skip. A small synthetic DataArray is used to avoid needing real atmospheric data, and the returned path must point to
        a file that exists on disk.

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

    def test_default_output_path(self, tmp_path: Path) -> None:
        """
        Verify that plot_horizontal_map uses a default output path containing 'horizontal_map.png' when none is given. When the caller does not provide an explicit output_path, the method should construct one based on the configured plot directory. This ensures map plots always land in a predictable location even when called without a fully specified path.

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


# -----------------------------------------------------------------------
# Helpers for visualizer gap tests
# -----------------------------------------------------------------------

def _write_csv(path: Path, rows: list) -> None:
    """
    Write a list of row dictionaries as a CSV file at the given path. This helper is used by gap-closing visualizer tests to create minimal CSV fixtures with specific domain or threshold values to trigger filtered-empty branches. The DataFrame is written with index=False to match the format expected by Visualizer.read_csv_data.

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
    Return a minimal ModvxConfig rooted in tmp_path with all standard sub-directory keys set. The configuration points fcst_dir, obs_dir, output_dir, csv_dir, and plot_dir to simple relative sub-directory names under tmp_path. This helper avoids repeating the same ModvxConfig construction boilerplate in every gap-closing visualizer test.

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


# -----------------------------------------------------------------------
# Empty / filtered data branches
# -----------------------------------------------------------------------


class TestVisualizerEmptyFilteredData:
    """Lines 122-123: filtered data empty → warning + continue."""

    def test_skip_when_filtered_empty(self, tmp_path: Path) -> None:
        """
        Verify that plot_fss_vs_leadtime logs a warning and skips the combination when filtered data is empty. The CSV contains only a row for domain 'OTHER', so requesting domain 'GLOBAL' yields an empty DataFrame. The method should emit a logging warning containing 'No data for' and continue rather than raising an exception, ensuring the full batch loop is not aborted by a missing combination.

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
        with patch.object(logging.getLogger("modvx.visualizer"), "warning") as mock_warn:
            viz.plot_fss_vs_leadtime(
                domain="GLOBAL", thresh="90", window="3",
                csv_dir=str(csv_dir), output_dir=str(tmp_path / "plots"),
            )
        assert any("No data for" in str(c) for c in mock_warn.call_args_list)


class TestVisualizerDiffEmptyControl:
    """Lines 242-243: control data empty in plot_fss_difference → return None."""

    def test_empty_control_returns_none(self, tmp_path: Path) -> None:
        """
        Verify that plot_fss_difference returns None when the control CSV contains no data matching the filter. The control CSV has only domain 'OTHER', so filtering for 'GLOBAL' yields an empty DataFrame. In this situation the difference cannot be computed and None must be returned rather than producing an empty or misleading plot.

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
    """Line 261: experiment filtered empty → continue past it."""

    def test_skip_empty_experiment(self, tmp_path: Path) -> None:
        """
        Verify that plot_fss_difference continues past an experiment CSV with no data matching the filter. The experiment CSV contains only domain 'OTHER' while control has 'GLOBAL', so the diff loop finds no experiment rows and skips the combination. The method should still return a valid result path from the control data rather than returning None.

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
    """Lines 324-326: cartopy import fails → return None."""

    def test_no_cartopy_returns_none(self, tmp_path: Path) -> None:
        """
        Verify that plot_horizontal_map returns None when the cartopy import fails at runtime. The method should catch the ImportError raised by the mocked import and return None rather than propagating the exception. This ensures deployments without Cartopy do not crash during map-plot attempts in generate_all_plots or CLI runs.

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

        with patch("builtins.__import__", side_effect=_mock_import):
            result = viz.plot_horizontal_map(
                field=xr.DataArray(np.zeros((2, 2)), dims=["latitude", "longitude"]),
                title="test",
                output_path=str(tmp_path / "map.png"),
            )
        assert result is None


class TestVisualizerGenerateAllPlotsException:
    """Lines 418-419: exception in plot_fss_vs_leadtime during generate_all_plots."""

    def test_exception_counted(self, tmp_path: Path) -> None:
        """
        Verify that generate_all_plots returns 0 when plot_fss_vs_leadtime raises an exception for every combination. This simulates a scenario where all individual plot calls fail, for example due to a rendering error. The batch method must catch the exception, decrement or skip the count, and return 0 rather than propagating the exception to the caller.

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

        with patch.object(viz, "plot_fss_vs_leadtime", side_effect=RuntimeError("boom")):
            count = viz.generate_all_plots(
                csv_dir=str(csv_dir), output_dir=str(plot_dir),
            )
        assert count == 0
