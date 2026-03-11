#!/usr/bin/env python3

"""
Visualisation helpers for modvx plotting utilities.

This module provides the :class:`Visualizer` class and supporting helpers to
produce verification plots from accumulated CSV results. It supports
metric-vs-lead-time comparison plots, metric-difference plots (experiment −
control), Cartopy horizontal maps for gridded fields, and batch generation
across domains, thresholds and windows. By default plot outputs are written
as PNG files to the configured plot directory.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
from itertools import product
from pathlib import Path
from typing import Any, cast, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .config import ModvxConfig

logger = logging.getLogger(__name__)

# Matplotlib imported at function level to support headless environments.

_ALL_METRICS = ["fss", "pod", "far", "csi", "fbias", "ets"]

_METRIC_LABELS: dict[str, str] = {
    "fss": "Fractions Skill Score (FSS)",
    "pod": "Probability of Detection (POD)",
    "far": "False Alarm Ratio (FAR)",
    "csi": "Critical Success Index (CSI)",
    "fbias": "Frequency Bias (FBIAS)",
    "ets": "Equitable Threat Score (ETS)",
}

# Metrics bounded within [0, 1]; others get dynamic y-limits.
_BOUNDED_METRICS = {"fss", "pod", "far", "csi"}


class Visualizer:
    """
    The :class:`Visualizer` centralises plotting logic for verification metrics computed from per-experiment CSV outputs. It supports per-metric comparison plots across experiments, experiment-minus-control difference plots, optional Cartopy-based horizontal maps of gridded fields, and batch generation of combinations discovered in CSV data. All outputs are saved as high-resolution PNG files in the configured plotting directory.

    Parameters:
        config (ModvxConfig): Run configuration providing directory paths
            and experiment settings used by plotting methods.
    """

    def __init__(self, config: ModvxConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Shared plot helpers
    # ------------------------------------------------------------------

    def _resolve_plot_dirs(
        self,
        csv_dir: Optional[str],
        output_dir: Optional[str],
    ) -> Tuple[str, str]:
        """
        This helper determines the CSV input directory and the output directory for plots by falling back to values defined in the associated :class:`ModvxConfig` when explicit arguments are not provided. It also ensures the resolved output directory exists on disk so callers may safely write files to it.

        Parameters:
            csv_dir (str, optional): Optional CSV directory path to override
                the configuration value. If ``None`` the config value is used.
            output_dir (str, optional): Optional output directory path to
                override the configuration value. If ``None`` the config
                value is used.

        Returns:
            Tuple[str, str]: A tuple ``(resolved_csv, resolved_out)`` with both
            paths as resolved strings.
        """
        cfg = self.config
        resolved_csv = csv_dir or cfg.resolve_path(cfg.csv_dir)
        resolved_out = output_dir or cfg.resolve_path(cfg.plot_dir)
        os.makedirs(resolved_out, exist_ok=True)
        return resolved_csv, resolved_out

    @staticmethod
    def _filter_df(
        results_df: pd.DataFrame,
        domain: str,
        thresh_val: float,
        window_val: int,
    ) -> pd.DataFrame:
        """
        This static helper applies boolean filtering on the provided DataFrame to select rows that match the specified domain name, percentile threshold, and neighbourhood window size. It preserves the original row ordering and returns a DataFrame containing only the matching rows (may be empty).

        Parameters:
            results_df (pandas.DataFrame): Input results DataFrame containing columns
                ``domain``, ``thresh``, ``window`` and metric columns.
            domain (str): Domain name to filter (for example, ``'GLOBAL'``).
            thresh_val (float): Threshold percentile value used to filter
                rows (e.g. ``90.0``).
            window_val (int): Neighbourhood window size used to filter rows
                (e.g. ``3``).

        Returns:
            pandas.DataFrame: Filtered DataFrame containing only matching rows.
        """
        mask = (
            (results_df["domain"] == domain)
            & (results_df["thresh"] == thresh_val)
            & (results_df["window"] == window_val)
        )
        return cast(pd.DataFrame, results_df.loc[mask])

    @staticmethod
    def _aggregate_metric(
        csv_file: Path,
        domain: str,
        thresh_val: float,
        window_val: int,
        metric: str,
    ) -> Optional["pd.Series[float]"]:
        """
        This helper reads a single per-experiment CSV file, validates the requested metric column exists, filters rows to the specified domain, threshold and window, and computes the mean metric value for each lead time. It returns a :class:`pandas.Series` indexed by ``leadTime`` with float values, or ``None`` if the metric column is missing or no rows match the filter (warnings are logged in those cases).

        Parameters:
            csv_file (pathlib.Path): Path to the experiment CSV file to read.
            domain (str): Domain name to filter (e.g., ``'GLOBAL'``).
            thresh_val (float): Threshold percentile value used to filter rows.
            window_val (int): Neighbourhood window size used to filter rows.
            metric (str): Column name of the metric to aggregate (e.g., ``'fss'``).

        Returns:
            Optional[pandas.Series[float]]: Series of mean metric values indexed
                by ``leadTime`` (sorted), or ``None`` when the metric is not
                present or no rows match the filter criteria.
        """
        results_df = pd.read_csv(csv_file)
        if metric not in results_df.columns:
            logger.warning("Metric '%s' not in %s — skipping", metric, csv_file.name)
            return None
        filtered_df = Visualizer._filter_df(results_df, domain, thresh_val, window_val)
        if filtered_df.empty:
            logger.warning(
                "No data for %s with domain=%s thresh=%s win=%s",
                csv_file.stem, domain, thresh_val, window_val,
            )
            return None
        return cast("pd.Series[float]", filtered_df.groupby("leadTime")[metric].mean().sort_index())

    @staticmethod
    def _set_y_limits(ax: "Any", values_list: list[float], metric: str) -> None:
        """
        This function computes sensible y-axis limits from the finite numeric values in ``values``. A small padding is added to the data range to avoid clipping, and metrics known to be bounded (for example FSS, POD, FAR, CSI) are clamped to the interval [0, 1]. If ``values`` contains no finite numbers the function returns without modifying the axis.

        Parameters:
            ax (Any): Matplotlib Axes-like object on which to set y-limits.
            values (list of float): Sequence of metric values used to
                determine limits.
            metric (str): Metric name used to determine whether to clamp to
                the [0, 1] interval.

        Returns:
            None: The function modifies the axis in-place and returns None.
        """
        arr = np.asarray(values_list, dtype=float)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return
        rng = float(valid.max() - valid.min())
        pad = rng * 0.1 if rng > 0 else 0.05
        ymin = float(valid.min()) - pad
        ymax = float(valid.max()) + pad
        if metric in _BOUNDED_METRICS:
            ymin = max(0.0, ymin)
            ymax = min(1.0, ymax)
        ax.set_ylim(ymin, ymax)

    @staticmethod
    def _finalise_plot(
        fig: "Any",
        ax: "Any",
        xlabel: str,
        ylabel: str,
        title: str,
        out_path: str,
    ) -> str:
        """
        This helper applies axis labels, title, legend and grid styling, tightens the layout, writes the figure to ``out_path`` at high resolution, and closes the figure to release resources. It returns the path to the saved file for callers that wish to log or further manipulate the output path.

        Parameters:
            fig (Any): Matplotlib Figure object to finalize and save.
            ax (Any): Matplotlib Axes object associated with the figure.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            title (str): Plot title string.
            out_path (str): Destination file path for the saved PNG.

        Returns:
            str: The path to the saved PNG file (``out_path``).
        """
        import matplotlib.pyplot as plt

        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # FSS vs lead-time
    # ------------------------------------------------------------------

    def plot_fss_vs_leadtime(
        self,
        domain: str = "GLOBAL",
        thresh: str = "90",
        window: str = "3",
        csv_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        metric: str = "fss",
    ) -> Optional[str]:
        """
        This method reads per-experiment CSV files from ``csv_dir``, filters rows to the specified ``domain``, ``thresh`` and ``window``, computes the mean metric per lead time, and overlays a line for each experiment in a single figure. The figure is saved to ``output_dir`` using a filename convention that encodes metric, domain, threshold and window. If no CSV files are present or no matching rows are found the method returns ``None``.

        Parameters:
            domain (str): Verification domain name to filter by (e.g. "GLOBAL").
            thresh (str): Percentile threshold string to filter by (e.g. "90").
            window (str): Neighbourhood window size string to filter by (e.g. "3").
            csv_dir (str, optional): Directory containing per-experiment CSV
                files; defaults to the configured csv_dir when ``None``.
            output_dir (str, optional): Directory where the PNG will be saved;
                defaults to the configured plot_dir when ``None``.
            metric (str): Column name of the metric to plot (default "fss").

        Returns:
            Optional[str]: Path to the saved PNG file, or ``None`` when no
                CSV files or matching data are found.
        """
        import matplotlib.pyplot as plt

        csv_dir, output_dir = self._resolve_plot_dirs(csv_dir, output_dir)
        csv_files = sorted(Path(csv_dir).glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", csv_dir)
            return None

        thresh_val = float(thresh)
        window_val = int(window)
        metric_label = _METRIC_LABELS.get(metric, metric.upper())

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(csv_files)))
        all_values: list[float] = []

        for file_idx, csv_file in enumerate(csv_files):
            experiment_series = self._aggregate_metric(csv_file, domain, thresh_val, window_val, metric)
            if experiment_series is None:
                continue
            mean_values = experiment_series.to_numpy(dtype=float)
            all_values.extend(mean_values.tolist())
            ax.plot(
                experiment_series.index.to_numpy(), mean_values,
                marker="o", linewidth=2.5, markersize=8,
                label=csv_file.stem, color=colors[file_idx],
            )

        if all_values:
            self._set_y_limits(ax, all_values, metric)

        fname = f"{metric}_leadtime_{domain}_thresh{thresh}percent_window{window}.png"
        out = os.path.join(output_dir, fname)
        self._finalise_plot(
            fig, ax,
            xlabel="Lead Time (hours)",
            ylabel=metric_label,
            title=f"{metric_label} vs Lead Time | Domain: {domain}, Threshold: {thresh}%, Window: {window}",
            out_path=out,
        )
        logger.info("Saved plot → %s", out)
        return out

    # ------------------------------------------------------------------
    # FSS difference plots
    # ------------------------------------------------------------------

    def plot_fss_difference(
        self,
        control_experiment: str,
        domain: str = "GLOBAL",
        thresh: str = "90",
        window: str = "3",
        csv_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        metric: str = "fss",
    ) -> Optional[str]:
        """
        This method computes the mean metric per lead time for the specified control experiment and then computes signed differences for every other experiment at common lead times. Difference curves are plotted and saved as a PNG file. If CSVs are missing or the control file has no matching rows the method returns ``None``.

        Parameters:
            control_experiment (str): Name of the baseline experiment (its
                CSV file must exist in ``csv_dir``).
            domain (str): Verification domain name to filter by (e.g. "GLOBAL").
            thresh (str): Percentile threshold string to filter by (e.g. "90").
            window (str): Neighbourhood window size string to filter by (e.g. "3").
            csv_dir (str, optional): Directory containing per-experiment CSV
                files; defaults to the configured csv_dir when ``None``.
            output_dir (str, optional): Directory where the PNG will be saved;
                defaults to the configured plot_dir when ``None``.
            metric (str): Column name of the metric to plot (default "fss").

        Returns:
            Optional[str]: Path to the saved difference plot PNG, or ``None``
                when the control file is missing or contains no matching data.
        """
        import matplotlib.pyplot as plt

        csv_dir, output_dir = self._resolve_plot_dirs(csv_dir, output_dir)
        csv_files = sorted(Path(csv_dir).glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", csv_dir)
            return None

        metric_label = _METRIC_LABELS.get(metric, metric.upper())
        thresh_val = float(thresh)
        window_val = int(window)

        # Load control
        control_path = Path(csv_dir) / f"{control_experiment}.csv"
        if not control_path.exists():
            logger.warning("Control CSV not found: %s", control_path)
            return None
        control_series = self._aggregate_metric(control_path, domain, thresh_val, window_val, metric)
        if control_series is None:
            logger.warning("No control data for domain=%s thresh=%s win=%s", domain, thresh, window)
            return None

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(csv_files)))

        for file_idx, csv_file in enumerate(csv_files):
            if csv_file.stem == control_experiment:
                continue
            experiment_series = self._aggregate_metric(csv_file, domain, thresh_val, window_val, metric)
            if experiment_series is None:
                continue
            common_index = control_series.index.intersection(experiment_series.index)
            difference_series = experiment_series.loc[common_index] - control_series.loc[common_index]
            ax.plot(
                common_index.to_numpy(), difference_series.to_numpy(dtype=float),
                marker="o", linewidth=2.5, markersize=8,
                label=csv_file.stem, color=colors[file_idx],
            )

        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        fname = f"{metric}_diff_{domain}_thresh{thresh}percent_window{window}.png"
        out = os.path.join(output_dir, fname)
        self._finalise_plot(
            fig, ax,
            xlabel="Lead Time (hours)",
            ylabel=f"Δ{metric_label} (exp − {control_experiment})",
            title=f"{metric_label} Difference | Domain: {domain}, Threshold: {thresh}%, Window: {window}",
            out_path=out,
        )
        logger.info("Saved difference plot → %s", out)
        return out

    # ------------------------------------------------------------------
    # Cartopy horizontal maps (optional)
    # ------------------------------------------------------------------

    def plot_horizontal_map(
        self,
        field: xr.DataArray,
        title: str = "",
        output_path: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        This method displays a two-dimensional :class:`xarray.DataArray` on a PlateCarree map with coastline, country borders and gridlines. If Cartopy is not installed the function logs a warning and returns ``None`` rather than raising an :class:`ImportError`. The figure is saved to ``output_path`` or the configured plot directory and the saved path is returned.

        Parameters:
            field (xarray.DataArray): Two-dimensional precipitation or mask
                field with latitude and longitude coordinates.
            title (str): Title string for the plot (default: empty string).
            output_path (str, optional): Full path for the output PNG file;
                defaults to ``<plot_dir>/horizontal_map.png`` when ``None``.
            **kwargs: Additional keyword arguments passed to the xarray plot
                method (for example ``cmap``).

        Returns:
            Optional[str]: Path to the saved PNG file, or ``None`` if cartopy
                is not installed.
        """
        try:
            import cartopy.crs as ccrs  # type: ignore[import-untyped]
            import cartopy.feature as cfeature  # type: ignore[import-untyped]
            from cartopy.mpl.geoaxes import GeoAxes  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("cartopy not installed — skipping horizontal map")
            return None

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(14, 8),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        geo_ax: GeoAxes = ax  # type: ignore[assignment]

        field.plot.pcolormesh(
            ax=geo_ax,
            transform=ccrs.PlateCarree(),
            cmap=kwargs.get("cmap", "viridis"),
            robust=True,
        )
        geo_ax.coastlines(linewidth=0.6)
        geo_ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
        geo_ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        geo_ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        if output_path is None:
            output_path = os.path.join(
                self.config.resolve_path(self.config.plot_dir), "horizontal_map.png"
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved horizontal map → %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_all_plots(
        self,
        csv_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> int:
        """
        The method inspects the first available CSV to discover unique domains, thresholds and windows, then enumerates all combinations for the requested metrics and calls :meth:`plot_fss_vs_leadtime` for each combination. Failures for individual combinations are logged as warnings and do not abort the batch. The method returns the number of successfully generated and saved plots.

        Parameters:
            csv_dir (str, optional): Directory containing per-experiment CSV
                files; defaults to the configured csv_dir when ``None``.
            output_dir (str, optional): Directory where plots will be saved;
                defaults to the configured plot_dir when ``None``.
            metrics (list of str, optional): Metric names to plot; defaults to
                all supported metrics when ``None``.

        Returns:
            int: Number of plots successfully generated and saved.
        """
        csv_dir, _ = self._resolve_plot_dirs(csv_dir, output_dir)
        csv_files = sorted(Path(csv_dir).glob("*.csv"))
        if not csv_files:
            logger.warning("No CSVs found in %s", csv_dir)
            return 0

        sample_df = pd.read_csv(csv_files[0])
        if metrics is None:
            metrics = [m for m in _ALL_METRICS if m in sample_df.columns]

        domains = sorted(sample_df["domain"].unique().tolist())
        thresholds = sorted(sample_df["thresh"].unique().tolist())
        windows = sorted(sample_df["window"].unique().tolist())

        total = len(metrics) * len(domains) * len(thresholds) * len(windows)
        logger.info(
            "Generating %d plots for %d metrics × %d domains × %d thresholds × %d windows",
            total, len(metrics), len(domains), len(thresholds), len(windows),
        )

        count = 0
        for metric, domain, threshold, window in product(metrics, domains, thresholds, windows):
            try:
                self.plot_fss_vs_leadtime(
                    domain=str(domain), thresh=str(threshold), window=str(window),
                    csv_dir=csv_dir, output_dir=output_dir, metric=metric,
                )
                count += 1
            except Exception:
                logger.warning("Failed: metric=%s domain=%s thresh=%s win=%s",
                               metric, domain, threshold, window, exc_info=True)

        logger.info("Generated %d / %d plots", count, total)
        return count

    # ------------------------------------------------------------------
    # Discover available options
    # ------------------------------------------------------------------

    def list_available_options(
        self, csv_dir: Optional[str] = None,
    ) -> Tuple[Optional[List[str]], Optional[List[float]], Optional[List[int]]]:
        """
        This helper reads the first CSV file in ``csv_dir`` and extracts the unique values for the ``domain``, ``thresh`` and ``window`` columns. The returned lists are sorted and typed appropriately so callers can iterate deterministically when generating plots or validating available options. If no CSV files are found the function returns a tuple of ``(None, None, None)``.

        Parameters:
            csv_dir (str, optional): Directory containing per-experiment CSV
                files; defaults to the configured csv_dir when ``None``.

        Returns:
            Tuple[Optional[List[str]], Optional[List[float]], Optional[List[int]]]:
                Tuple of ``(domains, thresholds, windows)`` sorted in
                ascending order, or ``(None, None, None)`` when no CSV files
                exist.
        """
        csv_dir = csv_dir or self.config.resolve_path(self.config.csv_dir)
        csv_files = sorted(Path(csv_dir).glob("*.csv"))
        if not csv_files:
            return None, None, None

        sample_df = pd.read_csv(csv_files[0])
        domains = sorted(str(x) for x in sample_df["domain"].unique())
        thresholds = sorted(float(x) for x in sample_df["thresh"].unique())
        windows = sorted(int(x) for x in sample_df["window"].unique())
        return domains, thresholds, windows
