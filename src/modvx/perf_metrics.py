"""
Performance metrics for modvx.

``PerfMetrics`` implements the NaN-aware Fraction Skill Score (FSS)
pipeline together with standard contingency-table verification metrics
(POD, FAR, CSI/TS, FBIAS, ETS) and basic point-wise scores (RMSE, bias).

The FSS pipeline:
    1. Compute percentile thresholds for forecast and observation.
    2. Generate binary masks (≥ threshold → 1, else 0; NaN preserved).
    3. Compute NaN-aware fractional coverage via uniform filter.
    4. Calculate FSS = 1 − MSE / MSE_ref.

Contingency-table metrics:
    From paired binary masks a 2×2 contingency table (hits, misses,
    false alarms, correct negatives) is derived, enabling POD, FAR,
    CSI/TS, frequency bias (FBIAS), and Equitable Threat Score (ETS).
"""

from __future__ import annotations

import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from .config import ModvxConfig

logger = logging.getLogger(__name__)


class PerfMetrics:
    """
    Calculate spatial verification metrics for forecast precipitation fields.
    The primary metric implemented is the NaN-aware Fraction Skill Score (FSS), which
    quantifies the spatial accuracy of precipitation forecasts at varying neighbourhood
    scales and intensity thresholds. Contingency-table metrics (POD, FAR, CSI/TS, FBIAS,
    ETS) are also supported and computed from the same binary exceedance masks used by
    FSS. Additional point-wise metrics (RMSE, bias) are provided as static utility
    methods. The batch interface sweeps all (threshold × window) combinations in one pass.

    Parameters:
        config (ModvxConfig): Run configuration with threshold lists, window sizes,
            and threshold mode settings.
    """

    def __init__(self, config: ModvxConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Binary mask
    # ------------------------------------------------------------------

    @staticmethod
    def generate_binary_mask(
        data: Union[xr.DataArray, np.ndarray],
        threshold: float,
    ) -> xr.DataArray:
        """
        Create a binary 0/1/NaN exceedance mask from a precipitation field at a given threshold.
        Grid points where the value is greater than or equal to the threshold are set to 1.0,
        all other non-NaN points are set to 0.0, and NaN positions in the input are preserved
        as NaN in the output. Preserving NaNs is critical so that out-of-domain points do not
        silently contribute as zeros to FSS numerator or denominator calculations.

        Parameters:
            data (xr.DataArray or np.ndarray): Input precipitation field, possibly containing
                NaN values.
            threshold (float): Minimum value for exceedance; values >= threshold become 1.0,
                others become 0.0.

        Returns:
            xr.DataArray: Binary mask array (float32) with values of 0.0, 1.0, or NaN.
        """
        arr = data.values if isinstance(data, xr.DataArray) else np.asarray(data)
        mask = (arr >= threshold).astype(np.float32)
        mask[np.isnan(arr)] = np.nan

        if isinstance(data, xr.DataArray):
            return xr.DataArray(mask, coords=data.coords, dims=data.dims, attrs=data.attrs)
        return xr.DataArray(mask)

    # ------------------------------------------------------------------
    # Fractional coverage
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fractional_field(
        binary_mask: Union[xr.DataArray, np.ndarray],
        window_size: int,
    ) -> xr.DataArray:
        """
        Compute the local fraction of exceedance within a square neighbourhood for each grid point.
        A uniform (box-car) filter of side *window_size* is applied over the binary mask. The
        implementation is NaN-aware: the sum of valid values is divided by the count of valid
        neighbours rather than the full window area, preventing domain-edge dilution where parts
        of the window extend outside the verification region. NaN positions in the input are
        preserved as NaN in the output fractional field.

        Parameters:
            binary_mask (xr.DataArray or np.ndarray): Binary 0/1/NaN exceedance mask from
                generate_binary_mask.
            window_size (int): Side length of the square neighbourhood window in grid points.

        Returns:
            xr.DataArray: Fractional coverage field with values in [0, 1] or NaN for
                out-of-domain points.
        """
        arr = binary_mask.values if isinstance(binary_mask, xr.DataArray) else np.asarray(binary_mask)
        nan_mask = np.isnan(arr)

        if nan_mask.any():
            arr_filled = np.where(nan_mask, 0.0, arr)
            valid_ind = (~nan_mask).astype(np.float64)
            ws2 = window_size ** 2

            sum_field = uniform_filter(arr_filled.astype(np.float64), size=window_size, mode="constant") * ws2
            valid_count = uniform_filter(valid_ind, size=window_size, mode="constant") * ws2

            with np.errstate(invalid="ignore", divide="ignore"):
                filtered = np.where(valid_count > 0, sum_field / valid_count, np.nan)
            filtered[nan_mask] = np.nan
        else:
            filtered = uniform_filter(arr, size=window_size, mode="constant")

        if isinstance(binary_mask, xr.DataArray):
            return xr.DataArray(filtered, coords=binary_mask.coords, dims=binary_mask.dims, attrs=binary_mask.attrs)
        return xr.DataArray(filtered)

    # ------------------------------------------------------------------
    # FSS
    # ------------------------------------------------------------------

    def calculate_fss(
        self,
        forecast_da: Union[xr.DataArray, np.ndarray],
        observation_da: Union[xr.DataArray, np.ndarray],
        percentile_threshold: float,
        neighborhood_size: int,
        *,
        experiment_name: str = "",
        cycle_start: Optional[datetime.datetime] = None,
        valid_time: Optional[datetime.datetime] = None,
        save_intermediate: bool = False,
    ) -> float:
        """
        Calculate the Fraction Skill Score (FSS) for a single (threshold, window) combination.
        Percentile thresholds are computed independently for forecast and observation. Binary
        exceedance masks are generated, optionally saved as debug NetCDF files, and then used
        to compute fractional coverage fields via uniform filtering. FSS is defined as
        1 − MSE/MSE_ref, where MSE_ref is the worst-case MSE under no spatial skill;
        returns 0.0 when MSE_ref is zero to avoid division-by-zero.

        Parameters:
            forecast_da (xr.DataArray or np.ndarray): Co-located, masked forecast field.
            observation_da (xr.DataArray or np.ndarray): Co-located, masked observation field.
            percentile_threshold (float): Percentile (0–100) used for binary-mask generation.
            neighborhood_size (int): Square neighbourhood window side-length in grid points.
            experiment_name (str): Experiment name, used when writing intermediate files.
            cycle_start (datetime.datetime, optional): Cycle initialisation time for
                intermediate file naming.
            valid_time (datetime.datetime, optional): Valid time for intermediate file naming.
            save_intermediate (bool): If True and cycle_start/valid_time are provided,
                write debug binary-mask NetCDF files.

        Returns:
            float: FSS value in [0, 1] where 1.0 indicates a perfect forecast.
        """
        # Compute percentile thresholds
        if isinstance(forecast_da, xr.DataArray):
            fcst_q = float(forecast_da.quantile(percentile_threshold / 100.0).values)
        else:
            fcst_q = float(np.nanquantile(np.asarray(forecast_da), percentile_threshold / 100.0))

        if isinstance(observation_da, xr.DataArray):
            obs_q = float(observation_da.quantile(percentile_threshold / 100.0).values)
        else:
            obs_q = float(np.nanquantile(np.asarray(observation_da), percentile_threshold / 100.0))

        fcst_bin = self.generate_binary_mask(forecast_da, fcst_q)
        obs_bin = self.generate_binary_mask(observation_da, obs_q)

        # Optionally save binary masks
        if save_intermediate and cycle_start is not None and valid_time is not None:
            from .file_manager import FileManager

            fm = FileManager(self.config)
            fm.save_intermediate_binary(
                fcst_bin, obs_bin, cycle_start, valid_time, percentile_threshold,
            )

        fcst_frac = self.compute_fractional_field(fcst_bin, neighborhood_size)
        obs_frac = self.compute_fractional_field(obs_bin, neighborhood_size)

        diff_sq = (fcst_frac - obs_frac) ** 2
        mse = float(np.nanmean(diff_sq.values))

        ref_sq = fcst_frac ** 2 + obs_frac ** 2
        ref_mse = float(np.nanmean(ref_sq.values))

        fss = 1.0 - mse / ref_mse if ref_mse > 0 else 0.0

        logger.info(
            "FSS (Threshold: %.1f%%, Window: %d): %.6f",
            percentile_threshold,
            neighborhood_size,
            fss,
        )
        return fss

    # ------------------------------------------------------------------
    # Contingency table
    # ------------------------------------------------------------------

    @staticmethod
    def compute_contingency_table(
        fcst_binary: Union[xr.DataArray, np.ndarray],
        obs_binary: Union[xr.DataArray, np.ndarray],
    ) -> Dict[str, int]:
        """
        Derive a 2×2 contingency table from paired binary exceedance masks. Grid points
        where either the forecast or observation mask is NaN are excluded from all counts,
        ensuring out-of-domain cells do not inflate correct negatives or misses. The four
        categories — hits, misses, false alarms, and correct negatives — are computed on the
        remaining valid Boolean pairs. The total valid-point count is also returned for use
        in metrics that require the sample size (e.g. ETS random-hits adjustment).

        Parameters:
            fcst_binary (xr.DataArray or np.ndarray): Binary 0/1/NaN forecast exceedance
                mask from generate_binary_mask.
            obs_binary (xr.DataArray or np.ndarray): Binary 0/1/NaN observation exceedance
                mask from generate_binary_mask.

        Returns:
            Dict[str, int]: Dictionary with keys ``hits``, ``misses``, ``false_alarms``,
                ``correct_negatives``, and ``total``.
        """
        f = fcst_binary.values if isinstance(fcst_binary, xr.DataArray) else np.asarray(fcst_binary, dtype=np.float64)
        o = obs_binary.values if isinstance(obs_binary, xr.DataArray) else np.asarray(obs_binary, dtype=np.float64)

        valid = ~(np.isnan(f) | np.isnan(o))
        fv = f[valid].astype(bool)
        ov = o[valid].astype(bool)

        hits = int(np.sum(fv & ov))
        misses = int(np.sum(~fv & ov))
        false_alarms = int(np.sum(fv & ~ov))
        correct_negatives = int(np.sum(~fv & ~ov))
        total = int(valid.sum())

        return {
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_negatives": correct_negatives,
            "total": total,
        }

    # ------------------------------------------------------------------
    # Contingency-table metrics
    # ------------------------------------------------------------------

    @staticmethod
    def pod(table: Dict[str, int]) -> float:
        """
        Compute the Probability of Detection (POD), also known as the hit rate. POD measures
        the fraction of observed events that were correctly forecast and ranges from 0 (no
        events detected) to 1 (all events detected). It is insensitive to false alarms and
        should be used alongside FAR or CSI for a complete assessment. Returns NaN when there
        are no observed events (hits + misses = 0).

        Parameters:
            table (Dict[str, int]): Contingency table from compute_contingency_table.

        Returns:
            float: POD value in [0, 1] or NaN if no observed events exist.
        """
        denom = table["hits"] + table["misses"]
        return table["hits"] / denom if denom > 0 else float("nan")

    @staticmethod
    def far(table: Dict[str, int]) -> float:
        """
        Compute the False Alarm Ratio (FAR), the fraction of forecast events that did not
        occur in observations. FAR ranges from 0 (no false alarms) to 1 (all forecasts were
        false alarms) and penalises over-forecasting. It is insensitive to missed events and
        should be considered alongside POD for a balanced evaluation. Returns NaN when there
        are no forecast events (hits + false_alarms = 0).

        Parameters:
            table (Dict[str, int]): Contingency table from compute_contingency_table.

        Returns:
            float: FAR value in [0, 1] or NaN if no forecast events exist.
        """
        denom = table["hits"] + table["false_alarms"]
        return table["false_alarms"] / denom if denom > 0 else float("nan")

    @staticmethod
    def csi(table: Dict[str, int]) -> float:
        """
        Compute the Critical Success Index (CSI), also called the Threat Score (TS). CSI
        measures the ratio of correct event forecasts to the total number of occasions where
        the event was either forecast or observed, combining the information from both POD
        and FAR into a single score. It ranges from 0 (no skill) to 1 (perfect) and is
        commonly used for rare precipitation events. Returns NaN when there are no events
        in either forecast or observation (hits + misses + false_alarms = 0).

        Parameters:
            table (Dict[str, int]): Contingency table from compute_contingency_table.

        Returns:
            float: CSI value in [0, 1] or NaN if no events exist in either field.
        """
        denom = table["hits"] + table["misses"] + table["false_alarms"]
        return table["hits"] / denom if denom > 0 else float("nan")

    # Threat Score is mathematically identical to CSI.
    ts = csi

    @staticmethod
    def fbias(table: Dict[str, int]) -> float:
        """
        Compute the Frequency Bias (FBIAS), the ratio of the number of forecast events to
        the number of observed events. A value of 1.0 indicates no frequency bias, values
        greater than 1 indicate over-forecasting, and values less than 1 indicate
        under-forecasting. FBIAS does not measure spatial accuracy — a spatially displaced
        forecast can still have perfect FBIAS. Returns NaN when there are no observed events
        (hits + misses = 0).

        Parameters:
            table (Dict[str, int]): Contingency table from compute_contingency_table.

        Returns:
            float: FBIAS value (positive real) or NaN if no observed events exist.
        """
        denom = table["hits"] + table["misses"]
        return (table["hits"] + table["false_alarms"]) / denom if denom > 0 else float("nan")

    @staticmethod
    def ets(table: Dict[str, int]) -> float:
        """
        Compute the Equitable Threat Score (ETS), also known as the Gilbert Skill Score.
        ETS adjusts CSI by removing the contribution of hits expected by random chance,
        providing a fairer comparison across events with different base rates. It ranges
        from −1/3 to 1, where 0 indicates no skill relative to random. The random-hit
        correction is computed as hits_random = (hits + misses)(hits + false_alarms) / total.
        Returns NaN when the denominator is zero or when the total sample size is zero.

        Parameters:
            table (Dict[str, int]): Contingency table from compute_contingency_table.

        Returns:
            float: ETS value in [−1/3, 1] or NaN if the computation is undefined.
        """
        total = table["total"]
        if total == 0:
            return float("nan")
        hits = table["hits"]
        misses = table["misses"]
        false_alarms = table["false_alarms"]
        hits_random = (hits + misses) * (hits + false_alarms) / total
        denom = hits + misses + false_alarms - hits_random
        return (hits - hits_random) / denom if denom != 0 else float("nan")

    # ------------------------------------------------------------------
    # Batch metrics (compute once, iterate over threshold × window)
    # ------------------------------------------------------------------

    def compute_fss_batch(
        self,
        forecast_da: xr.DataArray,
        observation_da: xr.DataArray,
        thresholds: Optional[List[float]] = None,
        window_sizes: Optional[List[int]] = None,
        *,
        experiment_name: str = "",
        cycle_start: Optional[datetime.datetime] = None,
        valid_time: Optional[datetime.datetime] = None,
        save_intermediate: bool = False,
    ) -> Dict[Tuple[float, int], Dict[str, float]]:
        """
        Compute FSS and contingency-table metrics for every (threshold, window) combination.
        Binary masks are computed once per threshold and reused across all window sizes for
        that threshold, avoiding redundant percentile and masking work. Contingency-table
        metrics (POD, FAR, CSI, FBIAS, ETS) are window-independent and are therefore computed
        once per threshold then copied into each window entry. FSS is computed per
        (threshold, window) pair as it depends on the neighbourhood size. Results are returned
        as a dictionary keyed by ``(threshold, window_size)`` mapping to a metrics dictionary.

        Parameters:
            forecast_da (xr.DataArray): Co-located, masked forecast precipitation field.
            observation_da (xr.DataArray): Co-located, masked observation precipitation field.
            thresholds (list of float, optional): Percentile thresholds to evaluate;
                defaults to config.thresholds.
            window_sizes (list of int, optional): Window sizes to evaluate;
                defaults to config.window_sizes.
            experiment_name (str): Experiment name passed through for intermediate file naming.
            cycle_start (datetime.datetime, optional): Cycle start time for intermediate files.
            valid_time (datetime.datetime, optional): Valid time for intermediate files.
            save_intermediate (bool): Whether to save intermediate binary mask files.

        Returns:
            Dict[Tuple[float, int], Dict[str, float]]: Mapping of ``(threshold, window_size)``
                to a dictionary with keys ``fss``, ``pod``, ``far``, ``csi``, ``fbias``,
                ``ets``.
        """
        thresholds = thresholds or self.config.thresholds
        window_sizes = window_sizes or self.config.window_sizes

        results: Dict[Tuple[float, int], Dict[str, float]] = {}

        for thr in thresholds:
            # --- Percentile thresholds (once per threshold) ---------------
            if isinstance(forecast_da, xr.DataArray):
                fcst_q = float(forecast_da.quantile(thr / 100.0).values)
            else:
                fcst_q = float(np.nanquantile(np.asarray(forecast_da), thr / 100.0))

            if isinstance(observation_da, xr.DataArray):
                obs_q = float(observation_da.quantile(thr / 100.0).values)
            else:
                obs_q = float(np.nanquantile(np.asarray(observation_da), thr / 100.0))

            fcst_bin = self.generate_binary_mask(forecast_da, fcst_q)
            obs_bin = self.generate_binary_mask(observation_da, obs_q)

            # Optionally save binary masks (once per threshold)
            if save_intermediate and cycle_start is not None and valid_time is not None:
                from .file_manager import FileManager
                fm = FileManager(self.config)
                fm.save_intermediate_binary(
                    fcst_bin, obs_bin, cycle_start, valid_time, thr,
                )

            # --- Contingency-table metrics (window-independent) ----------
            table = self.compute_contingency_table(fcst_bin, obs_bin)
            ct_metrics = {
                "pod": self.pod(table),
                "far": self.far(table),
                "csi": self.csi(table),
                "fbias": self.fbias(table),
                "ets": self.ets(table),
            }

            # --- FSS per window size -------------------------------------
            for win in window_sizes:
                fcst_frac = self.compute_fractional_field(fcst_bin, win)
                obs_frac = self.compute_fractional_field(obs_bin, win)

                diff_sq = (fcst_frac - obs_frac) ** 2
                mse = float(np.nanmean(diff_sq.values))

                ref_sq = fcst_frac ** 2 + obs_frac ** 2
                ref_mse = float(np.nanmean(ref_sq.values))

                fss = 1.0 - mse / ref_mse if ref_mse > 0 else 0.0

                logger.info(
                    "FSS (Threshold: %.1f%%, Window: %d): %.6f | "
                    "POD=%.4f FAR=%.4f CSI=%.4f FBIAS=%.4f ETS=%.4f",
                    thr, win, fss,
                    ct_metrics["pod"], ct_metrics["far"],
                    ct_metrics["csi"], ct_metrics["fbias"],
                    ct_metrics["ets"],
                )

                results[(thr, win)] = {"fss": fss, **ct_metrics}

        return results

    # ------------------------------------------------------------------
    # Placeholder: additional metrics
    # ------------------------------------------------------------------

    @staticmethod
    def rmse(forecast: np.ndarray, observation: np.ndarray) -> float:
        """
        Compute the Root Mean Squared Error between forecast and observation, ignoring NaN.
        Both input arrays are cast to float64 before differencing to prevent precision issues
        with integer or float32 inputs. NaN values from out-of-domain masking are excluded
        from the mean via np.nanmean. This is a simple pointwise metric and does not account
        for spatial structure or scale the way FSS does.

        Parameters:
            forecast (np.ndarray): Forecast values array; must have the same shape as
                observation.
            observation (np.ndarray): Observation values array; must have the same shape as
                forecast.

        Returns:
            float: Root Mean Squared Error computed over all non-NaN paired grid points.
        """
        diff = np.asarray(forecast, dtype=np.float64) - np.asarray(observation, dtype=np.float64)
        return float(np.sqrt(np.nanmean(diff ** 2)))

    @staticmethod
    def bias(forecast: np.ndarray, observation: np.ndarray) -> float:
        """
        Compute the mean bias (forecast minus observation) over all non-NaN grid points.
        A positive bias indicates systematic over-prediction while a negative bias reflects
        under-prediction. Both arrays are cast to float64 before differencing and NaN values
        from out-of-domain masking are excluded via np.nanmean. This metric provides a
        simple check for systematic offsets independent of spatial skill.

        Parameters:
            forecast (np.ndarray): Forecast values array; must have the same shape as
                observation.
            observation (np.ndarray): Observation values array; must have the same shape as
                forecast.

        Returns:
            float: Mean forecast bias (forecast − observation) over all non-NaN paired
                grid points.
        """
        diff = np.asarray(forecast, dtype=np.float64) - np.asarray(observation, dtype=np.float64)
        return float(np.nanmean(diff))
