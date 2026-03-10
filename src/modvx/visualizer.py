"""
Visualisation module for modvx.

``Visualizer`` produces:

* **Metric vs lead-time** comparison plots across experiments for FSS, POD,
  FAR, CSI, FBIAS, and ETS.
* **Metric difference** plots (experiment − control).
* **Cartopy horizontal maps** of precipitation / binary masks (optional).
* **Batch generation** of all (metric × domain × threshold × window) combos.
"""

from __future__ import annotations

import logging
import os
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

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
    Generate FSS verification plots from accumulated CSV result files.
    Supported output types include FSS-vs-lead-time comparison plots across experiments,
    FSS difference plots (experiment minus control), optional Cartopy horizontal maps of
    precipitation fields, and batch generation of all (domain × threshold × window) combos.
    All plot outputs are saved as high-resolution PNG files to the configured plot directory.

    Parameters:
        config (ModvxConfig): Run configuration with directory paths and experiment settings.
    """

    def __init__(self, config: ModvxConfig) -> None:
        self.config = config

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
        Plot a mean verification metric versus lead time for every experiment found in the
        CSV directory. Each CSV file is treated as one experiment, and mean metric values
        are computed per lead time after filtering to the specified domain, threshold, and
        window combination. Supported metrics are fss, pod, far, csi, fbias, and ets. The
        resulting lines are colour-coded by experiment and overlaid in a single figure.
        Returns None when no CSV files or matching data are found.

        Parameters:
            domain (str): Verification domain name to filter by (e.g. ``"GLOBAL"``).
            thresh (str): Percentile threshold string to filter by (e.g. ``"90"``).
            window (str): Neighbourhood window size string to filter by (e.g. ``"3"``).
            csv_dir (str, optional): Directory containing per-experiment CSV files;
                defaults to the configured csv_dir.
            output_dir (str, optional): Directory where the PNG will be saved;
                defaults to the configured plot_dir.
            metric (str): Column name of the metric to plot (default ``"fss"``).

        Returns:
            str or None: Path to the saved PNG file, or None if no data was found.
        """
        import matplotlib.pyplot as plt

        cfg = self.config
        csv_dir = csv_dir or cfg.resolve_path(cfg.csv_dir)
        output_dir = output_dir or cfg.resolve_path(cfg.plot_dir)
        os.makedirs(output_dir, exist_ok=True)

        csv_files = sorted(Path(csv_dir).glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", csv_dir)
            return None

        metric_label = _METRIC_LABELS.get(metric, metric.upper())

        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(csv_files)))
        all_vals: list[float] = []

        thresh_val = float(thresh)
        window_val = int(window)

        for idx, csv_file in enumerate(csv_files):
            df = pd.read_csv(csv_file)
            experiment = csv_file.stem

            if metric not in df.columns:
                logger.warning("Metric '%s' not in %s — skipping", metric, csv_file.name)
                continue

            filtered = df[
                (df["domain"] == domain)
                & (df["thresh"] == thresh_val)
                & (df["window"] == window_val)
            ]
            if filtered.empty:
                logger.warning("No data for %s with domain=%s thresh=%s win=%s", experiment, domain, thresh, window)
                continue

            grouped = filtered.groupby("leadTime")[metric].agg(["mean", "std", "count"]).sort_index()
            lead_times = grouped.index.to_numpy()
            mean_vals = grouped["mean"].to_numpy(dtype=float)

            all_vals.extend(mean_vals.tolist())
            ax.plot(
                lead_times, mean_vals,
                marker="o", linewidth=2.5, markersize=8,
                label=experiment, color=colors[idx],
            )

        ax.set_xlabel("Lead Time (hours)", fontsize=14, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=14, fontweight="bold")
        ax.set_title(
            f"{metric_label} vs Lead Time | Domain: {domain}, Threshold: {thresh}%, Window: {window}",
            fontsize=16, fontweight="bold", pad=20,
        )
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

        if all_vals:
            arr = np.asarray(all_vals, dtype=float)
            valid = arr[np.isfinite(arr)]
            if len(valid) > 0:
                rng = float(valid.max() - valid.min())
                pad = rng * 0.1 if rng > 0 else 0.05
                ymin = float(valid.min()) - pad
                ymax = float(valid.max()) + pad
                if metric in _BOUNDED_METRICS:
                    ymin = max(0.0, ymin)
                    ymax = min(1.0, ymax)
                ax.set_ylim(ymin, ymax)

        ax.tick_params(labelsize=12)
        fig.tight_layout()

        fname = f"{metric}_leadtime_{domain}_thresh{thresh}percent_window{window}.png"
        out = os.path.join(output_dir, fname)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
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
        Plot metric difference (experiment minus control) versus lead time for all experiments.
        The control experiment's mean metric value is subtracted from each other experiment's
        mean at common lead times, producing signed difference curves. A horizontal zero line
        provides a visual reference for neutral skill relative to the control. Supported metrics
        are fss, pod, far, csi, fbias, and ets. Returns None when no CSV files are found or
        when the control experiment CSV is missing.

        Parameters:
            control_experiment (str): Name of the baseline experiment (its CSV must exist
                in csv_dir).
            domain (str): Verification domain name to filter by (e.g. ``"GLOBAL"``).
            thresh (str): Percentile threshold string to filter by (e.g. ``"90"``).
            window (str): Neighbourhood window size string to filter by (e.g. ``"3"``).
            csv_dir (str, optional): Directory containing per-experiment CSV files;
                defaults to the configured csv_dir.
            output_dir (str, optional): Directory where the PNG will be saved;
                defaults to the configured plot_dir.
            metric (str): Column name of the metric to plot (default ``"fss"``).

        Returns:
            str or None: Path to the saved difference plot PNG, or None if data was not found.
        """
        import matplotlib.pyplot as plt

        cfg = self.config
        csv_dir = csv_dir or cfg.resolve_path(cfg.csv_dir)
        output_dir = output_dir or cfg.resolve_path(cfg.plot_dir)
        os.makedirs(output_dir, exist_ok=True)

        csv_files = sorted(Path(csv_dir).glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", csv_dir)
            return None

        metric_label = _METRIC_LABELS.get(metric, metric.upper())
        thresh_val = float(thresh)
        window_val = int(window)

        # Load control
        ctrl_path = Path(csv_dir) / f"{control_experiment}.csv"
        if not ctrl_path.exists():
            logger.warning("Control CSV not found: %s", ctrl_path)
            return None
        ctrl_df = pd.read_csv(ctrl_path)
        ctrl_filt = ctrl_df[
            (ctrl_df["domain"] == domain)
            & (ctrl_df["thresh"] == thresh_val)
            & (ctrl_df["window"] == window_val)
        ]
        if ctrl_filt.empty:
            logger.warning("No control data for domain=%s thresh=%s win=%s", domain, thresh, window)
            return None
        ctrl_mean = ctrl_filt.groupby("leadTime")[metric].mean()

        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(csv_files)))

        for idx, csv_file in enumerate(csv_files):
            experiment = csv_file.stem
            if experiment == control_experiment:
                continue
            df = pd.read_csv(csv_file)
            filt = df[
                (df["domain"] == domain)
                & (df["thresh"] == thresh_val)
                & (df["window"] == window_val)
            ]
            if filt.empty:
                continue
            exp_mean = filt.groupby("leadTime")[metric].mean()
            common = ctrl_mean.index.intersection(exp_mean.index)
            diff = exp_mean.loc[common] - ctrl_mean.loc[common]
            ax.plot(
                common.to_numpy(), diff.to_numpy(),
                marker="o", linewidth=2.5, markersize=8,
                label=experiment, color=colors[idx],
            )

        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Lead Time (hours)", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"Δ{metric_label} (exp − {control_experiment})", fontsize=14, fontweight="bold")
        ax.set_title(
            f"{metric_label} Difference | Domain: {domain}, Threshold: {thresh}%, Window: {window}",
            fontsize=16, fontweight="bold", pad=20,
        )
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=12)
        fig.tight_layout()

        fname = f"{metric}_diff_{domain}_thresh{thresh}percent_window{window}.png"
        out = os.path.join(output_dir, fname)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
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
        Plot a latitude-longitude field on a Cartopy PlateCarree map projection.
        Coastlines, country borders, and grid lines with labels are added automatically.
        The cartopy library must be installed; the function returns None and logs a warning
        when it is not available rather than raising an ImportError. The output is saved
        as a PNG at the specified path or to the default plot directory.

        Parameters:
            field (xr.DataArray): Two-dimensional precipitation or mask field with latitude
                and longitude coordinates.
            title (str): Title string for the plot (default: empty string).
            output_path (str, optional): Full path for the output PNG file; defaults to
                ``<plot_dir>/horizontal_map.png``.
            **kwargs: Additional keyword arguments passed to the xarray plot method
                (e.g. ``cmap``).

        Returns:
            str or None: Path to the saved PNG file, or None if cartopy is not installed.
        """
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.mpl.geoaxes import GeoAxes
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
        Generate metric-vs-leadtime plots for every (metric, domain, threshold, window)
        combination found in CSV data. By default all six metrics (fss, pod, far, csi, fbias,
        ets) are plotted. The unique domains, thresholds, and windows are discovered by reading
        the first available CSV file, then all combinations are enumerated and each plot is
        generated via plot_fss_vs_leadtime. Failures on individual combinations are caught and
        logged as warnings rather than aborting the batch. Returns the count of successfully
        saved plots.

        Parameters:
            csv_dir (str, optional): Directory containing per-experiment CSV files;
                defaults to the configured csv_dir.
            output_dir (str, optional): Directory where plots will be saved;
                defaults to the configured plot_dir.
            metrics (list of str, optional): Metric names to plot; defaults to all six
                metrics (fss, pod, far, csi, fbias, ets).

        Returns:
            int: Number of plots successfully generated and saved.
        """
        csv_dir = csv_dir or self.config.resolve_path(self.config.csv_dir)
        csv_files = sorted(Path(csv_dir).glob("*.csv"))
        if not csv_files:
            logger.warning("No CSVs found in %s", csv_dir)
            return 0

        if metrics is None:
            df0 = pd.read_csv(csv_files[0])
            metrics = [m for m in _ALL_METRICS if m in df0.columns]
        else:
            df0 = pd.read_csv(csv_files[0])

        domains = sorted(df0["domain"].unique().tolist())
        thresholds = sorted(df0["thresh"].unique().tolist())
        windows = sorted(df0["window"].unique().tolist())

        total = len(metrics) * len(domains) * len(thresholds) * len(windows)
        logger.info(
            "Generating %d plots for %d metrics × %d domains × %d thresholds × %d windows",
            total, len(metrics), len(domains), len(thresholds), len(windows),
        )

        count = 0
        for met, dom, thr, win in product(metrics, domains, thresholds, windows):
            try:
                self.plot_fss_vs_leadtime(
                    domain=str(dom), thresh=str(thr), window=str(win),
                    csv_dir=csv_dir, output_dir=output_dir, metric=met,
                )
                count += 1
            except Exception:
                logger.warning("Failed: metric=%s domain=%s thresh=%s win=%s",
                               met, dom, thr, win, exc_info=True)

        logger.info("Generated %d / %d plots", count, total)
        return count

    # ------------------------------------------------------------------
    # Discover available options
    # ------------------------------------------------------------------

    def list_available_options(
        self, csv_dir: Optional[str] = None,
    ) -> Tuple[Optional[List[str]], Optional[List[float]], Optional[List[int]]]:
        """
        Discover and return the unique domains, thresholds, and window sizes present in CSV data.
        The first CSV file found in csv_dir is read to extract the set of unique values for
        each dimension. This is used by the validate subcommand to list available plot options
        and by generate_all_plots to enumerate the full combination set. Returns a three-tuple
        of None values when no CSV files exist.

        Parameters:
            csv_dir (str, optional): Directory containing per-experiment CSV files;
                defaults to the configured csv_dir.

        Returns:
            Tuple[list of str or None, list of float or None, list of int or None]: Tuple of
                ``(domains, thresholds, windows)`` sorted in ascending order, or
                ``(None, None, None)`` when no CSV files are found.
        """
        csv_dir = csv_dir or self.config.resolve_path(self.config.csv_dir)
        csv_files = sorted(Path(csv_dir).glob("*.csv"))
        if not csv_files:
            return None, None, None

        df = pd.read_csv(csv_files[0])
        domains = sorted(str(x) for x in df["domain"].unique())
        thresholds = sorted(float(x) for x in df["thresh"].unique())
        windows = sorted(int(x) for x in df["window"].unique())
        return domains, thresholds, windows
