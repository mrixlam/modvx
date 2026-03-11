#!/usr/bin/env python3

"""
Performance metrics for modvx.

This module defines the PerfMetrics class, which encapsulates the logic for computing performance metrics such as the Fractions Skill Score (FSS) across multiple forecast cycles, valid times, domains, thresholds, and window sizes. The compute_fss_batch method is designed to efficiently compute FSS for a batch of threshold-window combinations while minimizing redundant data loading. By centralizing metric computation in this class, we ensure consistent application of scoring algorithms and facilitate future extensions to additional metrics as needed. The class is used by the TaskManager to process each work unit, which may involve multiple threshold and window combinations for the same forecast and observation data, thus optimizing I/O operations across the verification workflow.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
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
    Calculate spatial verification metrics for forecast precipitation fields. The primary metric implemented is the NaN-aware Fraction Skill Score (FSS), which quantifies the spatial accuracy of precipitation forecasts at varying neighbourhood scales and intensity thresholds. Contingency-table metrics (POD, FAR, CSI/TS, FBIAS, ETS) are also supported and computed from the same binary exceedance masks used by FSS. Additional point-wise metrics (RMSE, bias) are provided as static utility methods. The batch interface sweeps all (threshold × window) combinations in one pass.

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
        Create a binary 0/1/NaN exceedance mask from a precipitation field at a given threshold. Grid points where the value is greater than or equal to the threshold are set to 1.0, all other non-NaN points are set to 0.0, and NaN positions in the input are preserved as NaN in the output. Preserving NaNs is critical so that out-of-domain points do not silently contribute as zeros to FSS numerator or denominator calculations.

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
        Compute the local fraction of exceedance within a square neighbourhood for each grid point. A uniform (box-car) filter of side *window_size* is applied over the binary mask. The implementation is NaN-aware: the sum of valid values is divided by the count of valid neighbours rather than the full window area, preventing domain-edge dilution where parts of the window extend outside the verification region. NaN positions in the input are preserved as NaN in the output fractional field.

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
            valid_indicator = (~nan_mask).astype(np.float64)
            window_area = window_size ** 2

            summed_field = uniform_filter(arr_filled.astype(np.float64), size=window_size, mode="constant") * window_area
            valid_neighbor_count = uniform_filter(valid_indicator, size=window_size, mode="constant") * window_area

            with np.errstate(invalid="ignore", divide="ignore"):
                fractional_field = np.where(valid_neighbor_count > 0, summed_field / valid_neighbor_count, np.nan)
            fractional_field[nan_mask] = np.nan
        else:
            fractional_field = uniform_filter(arr, size=window_size, mode="constant")

        if isinstance(binary_mask, xr.DataArray):
            return xr.DataArray(fractional_field, coords=binary_mask.coords, dims=binary_mask.dims, attrs=binary_mask.attrs)
        return xr.DataArray(fractional_field)

    # ------------------------------------------------------------------
    # FSS
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_percentile_thresholds(
        forecast_da: Union[xr.DataArray, np.ndarray],
        observation_da: Union[xr.DataArray, np.ndarray],
        percentile: float,
    ) -> Tuple[float, float]:
        """
        This helper computes the requested percentile for both the forecast and observation fields independently, returning a pair of numeric threshold values. Support is provided for both :class:`xarray.DataArray` and NumPy array inputs, using the appropriate quantile routine for each.

        Parameters:
            forecast_da (xarray.DataArray or numpy.ndarray): Forecast field.
            observation_da (xarray.DataArray or numpy.ndarray): Observation field.
            percentile (float): Percentile to compute (0–100).

        Returns:
            Tuple[float, float]: ``(fcst_q, obs_q)`` threshold values for forecast and observation.
        """
        def _get_quantile_value(arr: Union[xr.DataArray, np.ndarray]) -> float:
            if isinstance(arr, xr.DataArray):
                return float(arr.quantile(percentile / 100.0).values)
            return float(np.nanquantile(np.asarray(arr), percentile / 100.0))

        return _get_quantile_value(forecast_da), _get_quantile_value(observation_da)

    def _make_binary_masks(
        self,
        forecast_da: Union[xr.DataArray, np.ndarray],
        observation_da: Union[xr.DataArray, np.ndarray],
        percentile: float,
        *,
        save_intermediate: bool = False,
        cycle_start: Optional[datetime.datetime] = None,
        valid_time: Optional[datetime.datetime] = None,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Percentile thresholds are computed independently for forecast and observation, and the corresponding binary masks (1.0 for exceedance, 0.0 otherwise, with NaNs preserved) are returned. When requested, the method also writes debug NetCDF files containing the binary masks to the debug directory using :class:`FileManager`.

        Parameters:
            forecast_da (xarray.DataArray or numpy.ndarray): Forecast field.
            observation_da (xarray.DataArray or numpy.ndarray): Observation field.
            percentile (float): Percentile used to compute thresholds.
            save_intermediate (bool): If True, save binary masks to debug NetCDF.
            cycle_start (datetime.datetime, optional): Cycle start time for file naming.
            valid_time (datetime.datetime, optional): Valid time for file naming.

        Returns:
            Tuple[xarray.DataArray, xarray.DataArray]: ``(fcst_bin, obs_bin)`` binary masks.
        """
        forecast_threshold, observation_threshold = self._compute_percentile_thresholds(forecast_da, observation_da, percentile)
        forecast_binary_mask = self.generate_binary_mask(forecast_da, forecast_threshold)
        observation_binary_mask = self.generate_binary_mask(observation_da, observation_threshold)
        if save_intermediate and cycle_start is not None and valid_time is not None:
            from .file_manager import FileManager
            FileManager(self.config).save_intermediate_binary(
                forecast_binary_mask, observation_binary_mask, cycle_start, valid_time, percentile,
            )
        return forecast_binary_mask, observation_binary_mask

    @staticmethod
    def _fss_from_fractions(
        forecast_fraction_field: xr.DataArray, observation_fraction_field: xr.DataArray,
    ) -> float:
        """
        Given pre-computed fractional coverage fields for forecast and observation, this helper computes the mean-squared-error (MSE) and the reference MSE for no-skill, then returns the FSS defined as ``1 - MSE / MSE_ref``. A guard ensures that when ``MSE_ref`` is zero the function returns 0.0 to avoid division by zero.

        Parameters:
            fcst_frac (xarray.DataArray): Forecast fractional coverage field.
            obs_frac (xarray.DataArray): Observation fractional coverage field.

        Returns:
            float: FSS scalar value in [0, 1].
        """
        mean_sq_error = float(np.nanmean(((forecast_fraction_field - observation_fraction_field) ** 2).values))
        reference_mse = float(np.nanmean((forecast_fraction_field ** 2 + observation_fraction_field ** 2).values))
        return 1.0 - mean_sq_error / reference_mse if reference_mse > 0 else 0.0

    def _compute_contingency_metrics(
        self, forecast_binary_mask: xr.DataArray, observation_binary_mask: xr.DataArray,
    ) -> Dict[str, float]:
        """
        The function computes the contingency table and returns a dictionary containing POD, FAR, CSI, FBIAS and ETS derived from the hit/miss/false-alarm counts. The helper centralises these computations so that calling code can reuse the same contingency data across multiple window-size evaluations.

        Parameters:
            fcst_bin (xarray.DataArray): Forecast binary exceedance mask.
            obs_bin (xarray.DataArray): Observation binary exceedance mask.

        Returns:
            Dict[str, float]: Dictionary with keys ``pod``, ``far``, ``csi``,
                ``fbias``, and ``ets``.
        """
        contingency_table = self.compute_contingency_table(forecast_binary_mask, observation_binary_mask)
        return {
            "pod": self.pod(contingency_table),
            "far": self.far(contingency_table),
            "csi": self.csi(contingency_table),
            "fbias": self.fbias(contingency_table),
            "ets": self.ets(contingency_table),
        }

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
        Calculate the Fraction Skill Score (FSS) for a single (threshold, window) combination. Percentile thresholds are computed independently for forecast and observation. Binary exceedance masks are generated, optionally saved as debug NetCDF files, and then used to compute fractional coverage fields via uniform filtering. FSS is defined as 1 − MSE/MSE_ref, where MSE_ref is the worst-case MSE under no spatial skill; returns 0.0 when MSE_ref is zero to avoid division-by-zero.

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
        forecast_binary, observation_binary = self._make_binary_masks(
            forecast_da, observation_da, percentile_threshold,
            save_intermediate=save_intermediate,
            cycle_start=cycle_start, valid_time=valid_time,
        )
        forecast_fractional_field = self.compute_fractional_field(forecast_binary, neighborhood_size)
        observation_fractional_field = self.compute_fractional_field(observation_binary, neighborhood_size)
        fss_value = self._fss_from_fractions(forecast_fractional_field, observation_fractional_field)
        logger.info(
            "FSS (Threshold: %.1f%%, Window: %d): %.6f",
            percentile_threshold, neighborhood_size, fss_value,
        )
        return fss_value

    # ------------------------------------------------------------------
    # Contingency table
    # ------------------------------------------------------------------

    @staticmethod
    def compute_contingency_table(
        fcst_binary: Union[xr.DataArray, np.ndarray],
        obs_binary: Union[xr.DataArray, np.ndarray],
    ) -> Dict[str, int]:
        """
        Derive a 2×2 contingency table from paired binary exceedance masks. Grid points where either the forecast or observation mask is NaN are excluded from all counts, ensuring out-of-domain cells do not inflate correct negatives or misses. The four categories — hits, misses, false alarms, and correct negatives — are computed on the remaining valid Boolean pairs. The total valid-point count is also returned for use in metrics that require the sample size (e.g. ETS random-hits adjustment).

        Parameters:
            fcst_binary (xr.DataArray or np.ndarray): Binary 0/1/NaN forecast exceedance
                mask from generate_binary_mask.
            obs_binary (xr.DataArray or np.ndarray): Binary 0/1/NaN observation exceedance
                mask from generate_binary_mask.

        Returns:
            Dict[str, int]: Dictionary with keys ``hits``, ``misses``, ``false_alarms``,
                ``correct_negatives``, and ``total``.
        """
        fcst_arr = fcst_binary.values if isinstance(fcst_binary, xr.DataArray) else np.asarray(fcst_binary, dtype=np.float64)
        obs_arr = obs_binary.values if isinstance(obs_binary, xr.DataArray) else np.asarray(obs_binary, dtype=np.float64)

        valid_mask = ~(np.isnan(fcst_arr) | np.isnan(obs_arr))
        forecast_bool = fcst_arr[valid_mask].astype(bool)
        obs_bool = obs_arr[valid_mask].astype(bool)

        hits = int(np.sum(forecast_bool & obs_bool))
        misses = int(np.sum(~forecast_bool & obs_bool))
        false_alarms = int(np.sum(forecast_bool & ~obs_bool))
        correct_negatives = int(np.sum(~forecast_bool & ~obs_bool))
        total = int(valid_mask.sum())

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
        Compute the Probability of Detection (POD), also known as the hit rate. POD measures the fraction of observed events that were correctly forecast and ranges from 0 (no events detected) to 1 (all events detected). It is insensitive to false alarms and should be used alongside FAR or CSI for a complete assessment. Returns NaN when there are no observed events (hits + misses = 0).

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
        Compute the False Alarm Ratio (FAR), the fraction of forecast events that did not occur in observations. FAR ranges from 0 (no false alarms) to 1 (all forecasts were false alarms) and penalises over-forecasting. It is insensitive to missed events and should be considered alongside POD for a balanced evaluation. Returns NaN when there are no forecast events (hits + false_alarms = 0).

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
        Compute the Critical Success Index (CSI), also called the Threat Score (TS). CSI measures the ratio of correct event forecasts to the total number of occasions where the event was either forecast or observed, combining the information from both POD and FAR into a single score. It ranges from 0 (no skill) to 1 (perfect) and is commonly used for rare precipitation events. Returns NaN when there are no events in either forecast or observation (hits + misses + false_alarms = 0).

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
        Compute the Frequency Bias (FBIAS), the ratio of the number of forecast events to the number of observed events. A value of 1.0 indicates no frequency bias, values greater than 1 indicate over-forecasting, and values less than 1 indicate under-forecasting. FBIAS does not measure spatial accuracy — a spatially displaced forecast can still have perfect FBIAS. Returns NaN when there are no observed events (hits + misses = 0).

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
        Compute the Equitable Threat Score (ETS), also known as the Gilbert Skill Score. ETS adjusts CSI by removing the contribution of hits expected by random chance, providing a fairer comparison across events with different base rates. It ranges from −1/3 to 1, where 0 indicates no skill relative to random. The random-hit correction is computed as hits_random = (hits + misses)(hits + false_alarms) / total. Returns NaN when the denominator is zero or when the total sample size is zero.

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
        Compute FSS and contingency-table metrics for every (threshold, window) combination. Binary masks are computed once per threshold and reused across all window sizes for that threshold, avoiding redundant percentile and masking work. Contingency-table metrics (POD, FAR, CSI, FBIAS, ETS) are window-independent and are therefore computed once per threshold then copied into each window entry. FSS is computed per (threshold, window) pair as it depends on the neighbourhood size. Results are returned as a dictionary keyed by ``(threshold, window_size)`` mapping to a metrics dictionary.

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

        for threshold in thresholds:
            forecast_binary, observation_binary = self._make_binary_masks(
                forecast_da, observation_da, threshold,
                save_intermediate=save_intermediate,
                cycle_start=cycle_start, valid_time=valid_time,
            )
            contingency_metrics = self._compute_contingency_metrics(forecast_binary, observation_binary)

            for window_size in window_sizes:
                forecast_fractional_field = self.compute_fractional_field(forecast_binary, window_size)
                observation_fractional_field = self.compute_fractional_field(observation_binary, window_size)
                fss_value = self._fss_from_fractions(forecast_fractional_field, observation_fractional_field)
                logger.info(
                    "FSS (Threshold: %.1f%%, Window: %d): %.6f | "
                    "POD=%.4f FAR=%.4f CSI=%.4f FBIAS=%.4f ETS=%.4f",
                    threshold, window_size, fss_value,
                    contingency_metrics["pod"], contingency_metrics["far"],
                    contingency_metrics["csi"], contingency_metrics["fbias"], contingency_metrics["ets"],
                )
                results[(threshold, window_size)] = {"fss": fss_value, **contingency_metrics}

        return results

    # ------------------------------------------------------------------
    # Placeholder: additional metrics
    # ------------------------------------------------------------------

    @staticmethod
    def rmse(forecast: np.ndarray, observation: np.ndarray) -> float:
        """
        Compute the Root Mean Squared Error between forecast and observation, ignoring NaN. Both input arrays are cast to float64 before differencing to prevent precision issues with integer or float32 inputs. NaN values from out-of-domain masking are excluded from the mean via np.nanmean. This is a simple pointwise metric and does not account for spatial structure or scale the way FSS does.

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
        Compute the mean bias (forecast minus observation) over all non-NaN grid points. A positive bias indicates systematic over-prediction while a negative bias reflects under-prediction. Both arrays are cast to float64 before differencing and NaN values from out-of-domain masking are excluded via np.nanmean. This metric provides a simple check for systematic offsets independent of spatial skill.

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
