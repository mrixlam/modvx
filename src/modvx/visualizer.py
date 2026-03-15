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

import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from itertools import product
from typing import Any, cast, List, Optional, Tuple


from .config import ModvxConfig

logger = logging.getLogger(__name__)


# List of all supported metrics for discovery and batch generation
_ALL_METRICS = ["fss", "pod", "far", "csi", "fbias", "ets"]

# Human-friendly labels for supported metrics used in plot titles and axis labels
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

# Metrics that do not depend on neighbourhood window size
_WINDOW_INDEPENDENT_METRICS = {"pod", "far", "csi", "fbias", "ets"}


class Visualizer:
    """
    The :class:`Visualizer` centralises plotting logic for verification metrics computed from per-experiment CSV outputs. It supports per-metric comparison plots across experiments, experiment-minus-control difference plots, optional Cartopy-based horizontal maps of gridded fields, and batch generation of combinations discovered in CSV data. All outputs are saved as high-resolution PNG files in the configured plotting directory.

    Parameters:
        config (ModvxConfig): Run configuration providing directory paths and experiment settings used by plotting methods.
    """

    def __init__(self, config: ModvxConfig) -> None:
        self.config = config


    def _resolve_plot_dirs(
        self,
        csv_dir: Optional[str],
        output_dir: Optional[str],
    ) -> Tuple[str, str]:
        """
        This helper determines the CSV input directory and the output directory for plots by falling back to values defined in the associated :class:`ModvxConfig` when explicit arguments are not provided. It also ensures the resolved output directory exists on disk so callers may safely write files to it.

        Parameters:
            csv_dir (str, optional): Optional CSV directory path to override the configuration value. If ``None`` the config value is used.
            output_dir (str, optional): Optional output directory path to override the configuration value. If ``None`` the config value is used.

        Returns:
            Tuple[str, str]: A tuple ``(resolved_csv, resolved_out)`` with both paths as resolved strings.
        """
        # Access the configuration object for path resolution and defaults
        cfg = self.config

        # Resolve the CSV directory path using the provided argument or falling back to the configuration value
        resolved_csv = csv_dir or cfg.resolve_relative_path(cfg.csv_dir)

        # Resolve the output directory path using the provided argument or falling back to the configuration value
        resolved_out = output_dir or cfg.resolve_relative_path(cfg.plot_dir)

        # Ensure the output directory exists 
        os.makedirs(resolved_out, exist_ok=True)

        # Return the resolved CSV directory and output directory as a tuple 
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
            results_df (pandas.DataFrame): Input results DataFrame containing columns ``domain``, ``thresh``, ``window`` and metric columns.
            domain (str): Domain name to filter (for example, ``'GLOBAL'``).
            thresh_val (float): Threshold percentile value used to filter rows (e.g. ``90.0``).
            window_val (int): Neighbourhood window size used to filter rows (e.g. ``3``).

        Returns:
            pandas.DataFrame: Filtered DataFrame containing only matching rows.
        """
        # Create a boolean mask to select rows where 'domain', 'thresh' and 'window' match the specified values
        mask = (
            (results_df["domain"] == domain)
            & (results_df["thresh"] == thresh_val)
            & (results_df["window"] == window_val)
        )

        # Return the filtered DataFrame containing only rows that match the specified criteria 
        return cast(pd.DataFrame, results_df.loc[mask])


    @staticmethod
    def _aggregate_metric(
        csv_file: Path,
        domain: str,
        thresh_val: float,
        window_val: Optional[int],
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
            Optional[pandas.Series[float]]: Series of mean metric values indexed by ``leadTime`` (sorted), or ``None`` when the metric is not present or no rows match the filter criteria.
        """
        # Read the CSV file into a DataFrame using pandas
        results_df = pd.read_csv(csv_file)

        # Log a warning and return None if the requested metric column is not present in the CSV's DataFrame columns
        if metric not in results_df.columns:
            logger.warning("Metric '%s' not in %s — skipping", metric, csv_file.name)
            return None
        
        if metric in _WINDOW_INDEPENDENT_METRICS:
            # Filter without window for threshold-only metrics.
            mask = (
                (results_df["domain"] == domain)
                & (results_df["thresh"] == thresh_val)
            )
        else:
            # Window-dependent metrics require a window value.
            if window_val is None:
                logger.warning("Window required for metric '%s'", metric)
                return None
            mask = (
                (results_df["domain"] == domain)
                & (results_df["thresh"] == thresh_val)
                & (results_df["window"] == window_val)
            )

        # Drop rows where the metric is missing to avoid mixing file types.
        mask = mask & results_df[metric].notna()

        # Apply filtering to select rows matching the specified criteria.
        filtered_df = cast(pd.DataFrame, results_df.loc[mask])

        # Log a warning and return None if the filtered DataFrame is empty, indicating no matching rows for the specified criteria in this CSV file.
        if filtered_df.empty:
            logger.warning(
                "No data for %s with domain=%s thresh=%s win=%s",
                csv_file.stem, domain, thresh_val, window_val,
            )
            return None
        
        # Return a Series of mean metric values indexed by leadTime, sorted by leadTime for consistent plotting
        return cast("pd.Series[float]", filtered_df.groupby("leadTime")[metric].mean().sort_index())


    @staticmethod
    def _set_y_limits(ax: "Any", values_list: list[float], metric: str) -> None:
        """
        This function computes sensible y-axis limits from the finite numeric values in ``values``. A small padding is added to the data range to avoid clipping, and metrics known to be bounded (for example FSS, POD, FAR, CSI) are clamped to the interval [0, 1]. If ``values`` contains no finite numbers the function returns without modifying the axis.

        Parameters:
            ax (Any): Matplotlib Axes-like object on which to set y-limits.
            values (list of float): Sequence of metric values used to determine limits.
            metric (str): Metric name used to determine whether to clamp to the [0, 1] interval.

        Returns:
            None: The function modifies the axis in-place and returns None.
        """
        # Convert the input list of values to a NumPy array for efficient numerical operations
        arr = np.asarray(values_list, dtype=float)
        valid = arr[np.isfinite(arr)]

        # If there are no valid finite values, return without setting limits to avoid errors or misleading plots
        if len(valid) == 0:
            return
        
        # Calculate the range of valid values and add a padding of 10% of the range 
        rng = float(valid.max() - valid.min())
        pad = rng * 0.1 if rng > 0 else 0.05

        # Calculate preliminary y-limits with padding to ensure data points are not clipped at the edges of the plot
        ymin = float(valid.min()) - pad
        ymax = float(valid.max()) + pad

        # Clamp y-limits to [0, 1] for metrics that are known to be bounded within this interval
        if metric in _BOUNDED_METRICS:
            ymin = max(0.0, ymin)
            ymax = min(1.0, ymax)
        
        # Specify the y-axis limits on the provided axis object 
        ax.set_ylim(ymin, ymax)


    def _leadtime_tick_interval(self) -> int:
        """
        Determine the x-axis tick interval for lead-time plots based on the configured forecast length.

        Returns:
            int: Lead-time tick interval in hours.
        """
        # Determine the lead-time tick interval based on the forecast length configured in the Visualizer's associated ModvxConfig. 
        forecast_length_hours = self.config.forecast_length_hours

        # For short forecast lengths up to 12 hours, use a 1-hour interval
        if forecast_length_hours <= 12:
            return 1
        
        # For forecast lengths between 12 and 24 hours, use a 3-hour interval 
        if forecast_length_hours <= 24:
            return 3
        
        # For forecast lengths between 24 and 48 hours, use a 6-hour interval 
        if forecast_length_hours <= 48:
            return 6
        
        # Return a 12-hour interval for forecast lengths greater than 48 hours
        return 12


    def _apply_leadtime_ticks(self, ax: "Any", lead_times: list[float]) -> None:
        """
        Apply lead-time ticks to the x-axis using the configured interval.

        Parameters:
            ax (Any): Matplotlib Axes-like object on which to set x-ticks.
            lead_times (list[float]): Available lead-time values used to bound ticks.

        Returns:
            None
        """
        # If there are no lead times provided, return without setting ticks 
        if not lead_times:
            return

        # Determine the lead-time tick interval using the helper method 
        step = self._leadtime_tick_interval()

        # Calculate the minimum and maximum lead times from the provided list 
        min_lead = float(min(lead_times))
        max_lead = float(max(lead_times))

        # Calculate the starting tick position based on the minimum lead time and the step interval.
        start = step if min_lead > 0 else 0

        # Generate an array of tick positions starting from the calculated start, up to the maximum lead time, with the determined step interval.
        ticks = np.arange(start, max_lead + step, step)

        # Set the x-ticks on the provided axis object to the calculated tick positions
        ax.set_xticks(ticks)


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

        # Set axis labels and title with styling for better readability and emphasis
        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # Add a legend if there are any labeled lines in the plot
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="best", fontsize=10, framealpha=0.9)

        # Add a grid with light styling to improve readability without overwhelming the data points
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=12)
        fig.tight_layout()

        # Save the figure to the specified output path with high resolution 
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

        # Close the figure to free up memory resources
        plt.close(fig)

        # Return the output path for logging 
        return out_path


    def plot_fss_vs_leadtime(
        self,
        domain: str = "GLOBAL",
        thresh: str = "90",
        window: Optional[str] = "3",
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
            csv_dir (str, optional): Directory containing per-experiment CSV files; defaults to the configured csv_dir when ``None``.
            output_dir (str, optional): Directory where the PNG will be saved; defaults to the configured plot_dir when ``None``.
            metric (str): Column name of the metric to plot (default "fss").

        Returns:
            Optional[str]: Path to the saved PNG file, or ``None`` when no CSV files or matching data are found.
        """
        import matplotlib.pyplot as plt

        # Resolve the CSV input directory and output directory using the provided arguments or falling back to configuration values.
        csv_dir, output_dir = self._resolve_plot_dirs(csv_dir, output_dir)

        # Recursively search for CSV files in the specified directory and sort them for deterministic processing order.
        csv_files = sorted(Path(csv_dir).glob("*.csv"))

        # Log a warning and return None if no CSV files are found in the specified directory 
        if not csv_files:
            logger.warning("No CSV files found in %s", csv_dir)
            return None

        # Convert the threshold string to a float for filtering
        thresh_val = float(thresh)

        # Convert the window string to an integer for window-dependent metrics
        window_val: Optional[int] = None

        # Only convert the window string to an integer if the metric is not in the list of window-independent metrics. 
        if metric not in _WINDOW_INDEPENDENT_METRICS:
            # Log a warning and return None if the metric requires a window value but none is provided
            if window is None:
                logger.warning("Window is required for metric '%s'", metric)
                return None
            
            # Convert the window string to an integer
            window_val = int(window)

        # Get a human-friendly label for the metric to use in plot titles and axis labels
        metric_label = _METRIC_LABELS.get(metric, metric.upper())

        # Create a figure and axis for the metric vs lead time plot with a specified size for better visibility
        fig, ax = plt.subplots(figsize=(12, 7))

        # Generate unique colors for each experiment using a colormap
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(csv_files)))

        # Initialize lists to collect all metric values and lead times across experiments for dynamic axis limits and ticks
        all_values: list[float] = []
        lead_times: list[float] = []

        # Iterate over all CSV files, aggregate the metric series for each experiment and specified filtering criteria
        for file_idx, csv_file in enumerate(csv_files):
            # Aggregate the metric series for the current experiment and specified filtering criteria. 
            experiment_series = self._aggregate_metric(csv_file, domain, thresh_val, window_val, metric)

            # Skip this experiment if the metric column is missing 
            if experiment_series is None:
                continue

            # Convert the metric values to a NumPy array of floats for consistent numerical processing and plotting
            mean_values = experiment_series.to_numpy(dtype=float)

            # Collect lead times for consistent x-axis ticks
            lead_times.extend(experiment_series.index.to_numpy(dtype=float).tolist())

            # Extend the all_values list with the mean metric values from this experiment 
            all_values.extend(mean_values.tolist())

            # Plot the metric vs lead time for the current experiment using a line with markers
            ax.plot(
                experiment_series.index.to_numpy(), mean_values,
                marker="o", linewidth=2.5, markersize=8,
                label=csv_file.stem, color=colors[file_idx],
            )

        # Set y-axis limits based on the collected metric values across all experiments 
        if all_values:
            self._set_y_limits(ax, all_values, metric)

        # Apply lead-time ticks based on the collected lead times across all experiments 
        if lead_times:
            self._apply_leadtime_ticks(ax, lead_times)

        # Specify the filename for the plot using the metric, domain, threshold and (optional) window.
        if window_val is None:
            fname = f"{metric}_vs_leadtime_vxdomain_{domain.lower()}_thresh_pct{thresh.replace('.', 'p')}.png"
            title = f"{metric_label} vs Lead Time | Domain: {domain}, Threshold: {thresh}%"
        else:
            fname = (
                f"{metric}_vs_leadtime_vxdomain_{domain.lower()}_"
                f"thresh_pct{thresh.replace('.', 'p')}_nbhd_pts{window}.png"
            )
            title = (
                f"{metric_label} vs Lead Time | Domain: {domain}, "
                f"Threshold: {thresh}%, Window: {window}"
            )

        # Ensure the output directory exists before saving the figure
        out = os.path.join(output_dir, fname)

        # Generate the plot title and axis labels using the metric label and filtering criteria
        self._finalise_plot(
            fig, ax,
            xlabel="Lead Time (hours)",
            ylabel=metric_label,
            title=title,
            out_path=out,
        )

        # Log the path to the saved plot for user reference
        logger.info("Saved plot → %s", out)

        # Return the path to the saved plot 
        return out


    def plot_fss_difference(
        self,
        control_experiment: str,
        domain: str = "GLOBAL",
        thresh: str = "90",
        window: Optional[str] = "3",
        csv_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        metric: str = "fss",
    ) -> Optional[str]:
        """
        This method computes the mean metric per lead time for the specified control experiment and then computes signed differences for every other experiment at common lead times. Difference curves are plotted and saved as a PNG file. If CSVs are missing or the control file has no matching rows the method returns ``None``.

        Parameters:
            control_experiment (str): Name of the baseline experiment (its CSV file must exist in ``csv_dir``).
            domain (str): Verification domain name to filter by (e.g. "GLOBAL").
            thresh (str): Percentile threshold string to filter by (e.g. "90").
            window (str): Neighbourhood window size string to filter by (e.g. "3").
            csv_dir (str, optional): Directory containing per-experiment CSV files; defaults to the configured csv_dir when ``None``.
            output_dir (str, optional): Directory where the PNG will be saved; defaults to the configured plot_dir when ``None``.
            metric (str): Column name of the metric to plot (default "fss").

        Returns:
            Optional[str]: Path to the saved difference plot PNG, or ``None`` when the control file is missing or contains no matching data.
        """
        import matplotlib.pyplot as plt

        # Resolve the CSV input directory and output directory using the provided arguments or falling back to configuration values.
        csv_dir, output_dir = self._resolve_plot_dirs(csv_dir, output_dir)

        # Recursively search for CSV files in the specified directory and sort them for deterministic processing order.
        csv_files = sorted(Path(csv_dir).glob("*.csv"))

        # Log a warning and return None if no CSV files are found in the specified directory
        if not csv_files:
            logger.warning("No CSV files found in %s", csv_dir)
            return None

        # Get a human-friendly label for the metric to use in plot titles and axis labels
        metric_label = _METRIC_LABELS.get(metric, metric.upper())

        # Convert the threshold string to a float for filtering purposes
        thresh_val = float(thresh)

        # Convert the window string to an integer for window-dependent metrics
        window_val: Optional[int] = None

        # Only convert the window string to an integer if the metric is not in the list of window-independent metrics.
        if metric not in _WINDOW_INDEPENDENT_METRICS:
            # Log a warning and return None if the metric requires a window value but none is provided
            if window is None:
                logger.warning("Window is required for metric '%s'", metric)
                return None
            
            # Convert the window string to an integer for filtering purposes
            window_val = int(window)

        # Construct the expected path to the control experiment's CSV file
        control_path = Path(csv_dir) / f"{control_experiment}.csv"

        # Log a warning and return None if the control CSV file does not exist in the specified directory
        if not control_path.exists():
            logger.warning("Control CSV not found: %s", control_path)
            return None
        
        # Aggregate the control series for the specified domain, threshold and window to use as the baseline for difference computation.
        control_series = self._aggregate_metric(control_path, domain, thresh_val, window_val, metric)

        # Log a warning and return None if the control series is missing 
        if control_series is None:
            logger.warning("No control data for domain=%s thresh=%s win=%s", domain, thresh, window)
            return None

        # Create a figure and axis for the difference plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Generate unique colors for each experiment using a colormap
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(csv_files)))

        # Iterate over all CSV files to compute and plot differences
        for file_idx, csv_file in enumerate(csv_files):
            # Skip the control experiment itself since the difference would be zero
            if csv_file.stem == control_experiment:
                continue

            # Load experiment series and skip if missing or no matching data for the specified criteria
            experiment_series = self._aggregate_metric(csv_file, domain, thresh_val, window_val, metric)

            # Skip if the experiment series is missing or has no matching data for the specified criteria
            if experiment_series is None:
                continue

            # Identify common lead times between the control and experiment series to ensure valid difference computation
            common_index = control_series.index.intersection(experiment_series.index)

            # Compute the difference between the experiment and control series at common lead times
            difference_series = experiment_series.loc[common_index] - control_series.loc[common_index]

            # Generate the difference plot for the current experiment compared to the control
            ax.plot(
                common_index.to_numpy(), difference_series.to_numpy(dtype=float),
                marker="o", linewidth=2.5, markersize=8,
                label=csv_file.stem, color=colors[file_idx],
            )

        # Add a horizontal line at y=0 to indicate no difference between experiment and control
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)

        # Apply lead-time ticks based on the control series range
        self._apply_leadtime_ticks(ax, control_series.index.to_numpy(dtype=float).tolist())

        # Specify the filename for the difference plot
        if window_val is None:
            fname = f"{metric}_diff_{domain}_thresh{thresh}percent.png"
            title = f"{metric_label} Difference | Domain: {domain}, Threshold: {thresh}%"
        else:
            fname = f"{metric}_diff_{domain}_thresh{thresh}percent_nbhd_pts{window}.png"
            title = (
                f"{metric_label} Difference | Domain: {domain}, "
                f"Threshold: {thresh}%, Window: {window}"
            )

        # Ensure the output directory exists before saving the figure
        out = os.path.join(output_dir, fname)

        # Specify the plot title and axis labels
        self._finalise_plot(
            fig, ax,
            xlabel="Lead Time (hours)",
            ylabel=f"Δ{metric_label} (exp − {control_experiment})",
            title=title,
            out_path=out,
        )

        # Log the path to the saved difference plot for user reference
        logger.info("Saved difference plot → %s", out)

        # Return the path to the saved difference plot PNG file
        return out


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
            field (xarray.DataArray): Two-dimensional precipitation or mask field with latitude and longitude coordinates.
            title (str): Title string for the plot (default: empty string).
            output_path (str, optional): Full path for the output PNG file; defaults to ``<plot_dir>/horizontal_map.png`` when ``None``.
            **kwargs: Additional keyword arguments passed to the xarray plot method (for example ``cmap``).

        Returns:
            Optional[str]: Path to the saved PNG file, or ``None`` if cartopy is not installed.
        """
        try:
            import cartopy.crs as ccrs  # type: ignore[import-untyped]
            import cartopy.feature as cfeature  # type: ignore[import-untyped]
            from cartopy.mpl.geoaxes import GeoAxes  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("cartopy not installed — skipping horizontal map")
            return None

        import matplotlib.pyplot as plt

        # Create a figure and GeoAxes with PlateCarree projection for the horizontal map
        fig, ax = plt.subplots(
            figsize=(14, 8),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        # Cast the Axes to GeoAxes for type checking purposes 
        geo_ax: GeoAxes = ax  # type: ignore[assignment]

        # Generate horizontal map using xarray's built-in plotting method with Cartopy projection 
        field.plot.pcolormesh(
            ax=geo_ax,
            transform=ccrs.PlateCarree(),
            cmap=kwargs.get("cmap", "viridis"),
            robust=True,
        )

        # Add coastlines, borders and gridlines with styling for better visibility
        geo_ax.coastlines(linewidth=0.6)
        geo_ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
        geo_ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        geo_ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        # Determine the output path for the saved figure
        if output_path is None:
            output_path = os.path.join(
                self.config.resolve_relative_path(self.config.plot_dir), "horizontal_map.png"
            )
        
        # Ensure the output directory exists before saving the figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the figure to the specified output path with high resolution and tight layout
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        # Close the figure to release memory resources 
        plt.close(fig)

        # Log the path to the saved horizontal map for user reference
        logger.info("Saved horizontal map → %s", output_path)

        # Return the path to the saved horizontal map PNG file
        return output_path


    def generate_all_plots(
        self,
        csv_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> int:
        """
        The method inspects the first available CSV to discover unique domains, thresholds and windows, then enumerates all combinations for the requested metrics and calls :meth:`plot_fss_vs_leadtime` for each combination. Failures for individual combinations are logged as warnings and do not abort the batch. The method returns the number of successfully generated and saved plots.

        Parameters:
            csv_dir (str, optional): Directory containing per-experiment CSV files; defaults to the configured csv_dir when ``None``.
            output_dir (str, optional): Directory where plots will be saved; defaults to the configured plot_dir when ``None``.
            metrics (list of str, optional): Metric names to plot; defaults to all supported metrics when ``None``.

        Returns:
            int: Number of plots successfully generated and saved.
        """
        # Resolve the CSV input directory and output directory using the provided arguments or falling back to configuration values.
        csv_dir, _ = self._resolve_plot_dirs(csv_dir, output_dir)

        # Recursively search for CSV files in the specified directory and sort them for deterministic processing order.
        csv_files = sorted(Path(csv_dir).glob("*.csv"))

        # Log a warning and return 0 if no CSV files are found in the specified directory
        if not csv_files:
            logger.warning("No CSVs found in %s", csv_dir)
            return 0

        # Load the first CSV file to discover available domains, thresholds and windows for batch generation.
        sample_df = pd.read_csv(csv_files[0])

        # If no specific metrics are provided, default to all supported metrics that are present in the sample CSV's columns.
        if metrics is None:
            metrics = [m for m in _ALL_METRICS if m in sample_df.columns]

        # Extract the list of unique domain names as strings for consistent filename generation
        domains = sorted(sample_df["domain"].unique().tolist())

        # Extract the list of threshold values as strings for consistent filename generation
        thresholds = sorted(sample_df["thresh"].unique().tolist())

        # Extract the list of unique window values, dropping NaN from threshold-only rows
        windows = sorted(
            int(x) for x in sample_df["window"].dropna().unique().tolist()
        )

        # Compute the total number of plot combinations for logging purposes.
        window_dependent = [m for m in metrics if m not in _WINDOW_INDEPENDENT_METRICS]
        window_independent = [m for m in metrics if m in _WINDOW_INDEPENDENT_METRICS]

        # The total number of plots is the sum of window-dependent combinations and window-independent combinations 
        total = (
            len(window_dependent) * len(domains) * len(thresholds) * len(windows)
            + len(window_independent) * len(domains) * len(thresholds)
        )

        # Log the total number of plots that will be generated 
        logger.info(
            "Generating %d plots for %d metrics × %d domains × %d thresholds",
            total, len(metrics), len(domains), len(thresholds),
        )

        # Initialize a counter to track the number of successfully generated plots.
        count = 0

        # Iterate over all combinations of metrics, domains, thresholds and windows
        for metric, domain, threshold in product(metrics, domains, thresholds):
            if metric in _WINDOW_INDEPENDENT_METRICS:
                try:
                    self.plot_fss_vs_leadtime(
                        domain=str(domain), thresh=str(threshold), window=None,
                        csv_dir=csv_dir, output_dir=output_dir, metric=metric,
                    )
                    count += 1
                except Exception:
                    logger.warning("Failed: metric=%s domain=%s thresh=%s",
                                   metric, domain, threshold, exc_info=True)
            else:
                for window in windows:
                    try:
                        # Save the plot for the current combination and increment the count if successful.
                        self.plot_fss_vs_leadtime(
                            domain=str(domain), thresh=str(threshold), window=str(window),
                            csv_dir=csv_dir, output_dir=output_dir, metric=metric,
                        )
                        count += 1
                    except Exception:
                        logger.warning("Failed: metric=%s domain=%s thresh=%s win=%s",
                                       metric, domain, threshold, window, exc_info=True)

        # Log the total count of successfully generated plots out of the expected total combinations.
        logger.info("Generated %d / %d plots", count, total)

        # Return the count of successfully generated plots 
        return count


    def list_available_options(
        self, csv_dir: Optional[str] = None,
    ) -> Tuple[Optional[List[str]], Optional[List[float]], Optional[List[int]]]:
        """
        This helper reads the first CSV file in ``csv_dir`` and extracts the unique values for the ``domain``, ``thresh`` and ``window`` columns. The returned lists are sorted and typed appropriately so callers can iterate deterministically when generating plots or validating available options. If no CSV files are found the function returns a tuple of ``(None, None, None)``.

        Parameters:
            csv_dir (str, optional): Directory containing per-experiment CSV files; defaults to the configured csv_dir when ``None``.

        Returns:
            Tuple[Optional[List[str]], Optional[List[float]], Optional[List[int]]]: Tuple of ``(domains, thresholds, windows)`` sorted in ascending order, or ``(None, None, None)`` when no CSV files exist.
        """
        # Resolve the CSV directory path using the provided argument 
        csv_dir = csv_dir or self.config.resolve_relative_path(self.config.csv_dir)

        # Recursively search for CSV files in the specified directory 
        csv_files = sorted(Path(csv_dir).glob("*.csv"))

        # If no CSV files are found return None for all options to indicate no available data.
        if not csv_files:
            return None, None, None

        # Generate a dataframe from the first CSV file to extract unique values for domain, threshold and window. 
        sample_df = pd.read_csv(csv_files[0])

        # Extract the list of unique domain names as strings
        domains = sorted(str(x) for x in sample_df["domain"].unique())

        # Extract the list of threshold values as floats
        thresholds = sorted(float(x) for x in sample_df["thresh"].unique())

        # Extract the list of window values as integers, dropping NaN from threshold-only rows
        windows = sorted(int(x) for x in sample_df["window"].dropna().unique())

        # Return the discovered domain names, threshold values and window sizes as sorted lists 
        return domains, thresholds, windows
