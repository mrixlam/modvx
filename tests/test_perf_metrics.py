#!/usr/bin/env python3

"""
Unit tests for MODvx performance metrics.

This module contains a comprehensive suite of tests for the methods in the PerfMetrics class, which implements the core logic for computing FSS and contingency-table-based metrics. The tests cover a range of scenarios including perfect forecasts, random fields, NaN handling, and edge cases for each metric. By using synthetic data with known properties, these tests verify that the mathematical computations are correct and that the methods behave as expected under various conditions.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

import datetime
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from modvx.config import ModvxConfig
from modvx.perf_metrics import PerfMetrics


@pytest.fixture
def pm() -> PerfMetrics:
    """
    Construct a PerfMetrics instance backed by default ModvxConfig settings for use in FSS pipeline tests. This fixture provides a ready-to-use metrics object without requiring any filesystem access or YAML configuration files. Tests that call FSS, binary mask, and fractional field methods on real 2D grids rely on this fixture to avoid repeating boilerplate setup code.

    Returns:
        PerfMetrics: Fully initialised PerfMetrics object with default configuration.
    """
    return PerfMetrics(ModvxConfig())


class TestBinaryMask:
    """Tests for generate_binary_mask verifying threshold exceedance logic, NaN propagation, and support for both xarray and NumPy inputs."""

    def test_simple(self, pm: PerfMetrics) -> None:
        """
        Verify that generate_binary_mask correctly classifies values at and above the threshold as 1 and values below as 0. This test constructs a 1D array spanning 0.0 to 2.0 with a threshold of 1.0 and checks the output against an expected binary array. The threshold is inclusive, so a value exactly equal to the threshold should be classified as 1 in the output mask.

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        arr = xr.DataArray([0.0, 0.5, 1.0, 1.5, 2.0])
        mask = pm.generate_binary_mask(arr, 1.0)
        expected = np.array([0, 0, 1, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(mask.values, expected)

    def test_nan_preserved(self, pm: PerfMetrics) -> None:
        """
        Confirm that NaN values in the input array are preserved as NaN in the binary mask output rather than being classified as 0 or 1. This prevents undetected data gaps from being treated as valid below-threshold points, which would incorrectly inflate the count of zero-class cells during FSS fraction field computation. The test explicitly checks that non-NaN values around the NaN position are still classified correctly.
        
        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        arr = xr.DataArray([np.nan, 0.5, 1.5])
        mask = pm.generate_binary_mask(arr, 1.0)
        assert np.isnan(mask.values[0])
        assert mask.values[1] == 0.0
        assert mask.values[2] == 1.0

    def test_numpy_input(self, pm: PerfMetrics) -> None:
        """
        Verify that generate_binary_mask accepts a raw NumPy array as input and returns an xarray DataArray. Supporting NumPy inputs ensures the method can be called from code paths that have not yet converted arrays to xarray, without requiring the caller to wrap inputs manually. The returned type is explicitly checked to confirm the output is always a DataArray regardless of the input container type.

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        arr = np.array([0.0, 1.0, 2.0])
        mask = pm.generate_binary_mask(arr, 1.0)
        assert isinstance(mask, xr.DataArray)
        np.testing.assert_array_equal(mask.values, [0, 1, 1])


class TestFractionalField:
    """Tests for compute_fractional_field verifying neighborhood-averaged fraction values for uniform fields and correct NaN propagation through the sliding window."""

    def test_uniform_field(self, pm: PerfMetrics) -> None:
        """
        Verify that a uniformly all-ones binary field produces a fractional field of 1.0 at interior cells when passed through a window of size 3. Interior cells in an all-ones field have all neighborhood values equal to 1, so the window-averaged fraction must also be exactly 1.0. This test establishes the upper-bound baseline for fraction output before edge effects and partial-window cells are tested.
        
        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        arr = xr.DataArray(np.ones((11, 11)))
        frac = pm.compute_fractional_field(arr, 3)
        # Interior cells should be exactly 1
        assert float(frac.values[5, 5]) == pytest.approx(1.0)

    def test_nan_handling(self, pm: PerfMetrics) -> None:
        """
        Confirm that a NaN value in the binary field propagates through the sliding window and remains NaN at the corresponding position in the fractional field output. NaN positions represent missing or masked-out grid cells that should not contribute to or receive FSS fraction values, so their NaN state must be preserved rather than filled by neighborhood averaging. This ensures the fractional field correctly reflects the spatial extent of the verification mask applied upstream.

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        data = np.ones((5, 5))
        data[2, 2] = np.nan
        arr = xr.DataArray(data)
        frac = pm.compute_fractional_field(arr, 3)
        assert np.isnan(frac.values[2, 2])


class TestFSS:
    """Tests for calculate_fss verifying the FSS value bounds and the perfect-forecast degenerate case using synthetic 2D spatial fields."""

    def test_perfect_forecast(self, pm: PerfMetrics) -> None:
        """
        Verify that calculate_fss returns exactly 1.0 when forecast and observation fields are identical. A perfect forecast has zero mean squared error between the fractional fields, which by definition yields the maximum FSS score of 1.0. This test uses an identical random field for both inputs to confirm the degenerate case is handled correctly without any floating-point drift.

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        field = xr.DataArray(
            np.random.RandomState(42).rand(20, 20),
            dims=["latitude", "longitude"],
        )
        fss = pm.calculate_fss(field, field, 90.0, 3)
        assert fss == pytest.approx(1.0, abs=1e-6)

    def test_fss_range(self, pm: PerfMetrics) -> None:
        """
        Confirm that calculate_fss returns a value at or below 1.0 for uncorrelated random forecast and observation fields. While FSS can theoretically be negative for forecasts worse than random, it should never exceed 1.0 in any configuration. This upper-bound check uses independent random seeds to produce fields with no spatial correlation, representing a realistic worst-case verification scenario.

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        rng = np.random.RandomState(0)
        fcst = xr.DataArray(rng.rand(30, 30), dims=["latitude", "longitude"])
        obs = xr.DataArray(rng.rand(30, 30), dims=["latitude", "longitude"])
        fss = pm.calculate_fss(fcst, obs, 90.0, 5)
        assert fss <= 1.0


class TestRMSE:
    """Tests for the PerfMetrics.rmse static method verifying zero-error and known-value cases."""

    def test_zero_error(self) -> None:
        """
        Verify that rmse returns exactly 0.0 when forecast and observation arrays are identical. This is the theoretical minimum RMSE representing a perfect forecast with no error at any grid point. Confirming the zero-error case ensures the implementation does not introduce numerical bias or floating-point drift when computing the square root of a zero mean squared error.
        
        Returns: 
            None
        """
        a = np.array([1.0, 2.0, 3.0])
        assert PerfMetrics.rmse(a, a) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """
        Confirm that rmse computes the correct value of 1.0 for a forecast array of all zeros compared to an observation array of all ones. The mean squared error for this case is exactly 1.0, and its square root is also 1.0, providing a simple hand-verifiable reference. This known-value test catches sign errors, incorrect averaging, or missing square root steps in the RMSE formula.

        Returns:
            None
        """
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert PerfMetrics.rmse(a, b) == pytest.approx(1.0)


class TestBias:
    """Tests for the PerfMetrics.bias static method verifying zero-bias and positive-bias cases."""

    def test_no_bias(self) -> None:
        """
        Verify that bias returns exactly 0.0 when forecast and observation arrays are identical. A zero-bias result for identical inputs confirms the implementation correctly computes the mean difference and does not introduce any constant offset. This baseline case is required before testing the directional and signed behavior of the bias metric.

        Returns:
            None
        """
        a = np.array([1.0, 2.0, 3.0])
        assert PerfMetrics.bias(a, a) == pytest.approx(0.0)

    def test_positive_bias(self) -> None:
        """
        Confirm that bias returns a positive value of 1.0 when the forecast is uniformly 1.0 higher than the observation at every point. This verifies the sign convention and magnitude of the bias calculation — positive bias means the forecast systematically overpredicts relative to observations. Correct sign convention is critical for interpreting whether a model is wet-biased or dry-biased in precipitation verification.

        Returns:
            None
        """
        fcst = np.array([2.0, 3.0])
        obs = np.array([1.0, 2.0])
        assert PerfMetrics.bias(fcst, obs) == pytest.approx(1.0)


# ======================================================================
# Contingency table
# ======================================================================


class TestContingencyTable:
    """Tests for compute_contingency_table verifying correct hit/miss/false-alarm/correct-negative counts and NaN exclusion from all categories."""

    def test_known_counts(self) -> None:
        """
        Verify that compute_contingency_table produces the exact expected count for each category given a hand-crafted 2x2 input. The forecast has events at positions 0 and 1, the observation at positions 0 and 2, so position 0 is a hit, position 1 is a false alarm, position 2 is a miss, and position 3 is a correct negative. The total must equal the number of grid points since no NaNs are present.
        
        Returns:
            None
        """
        fcst = np.array([1.0, 1.0, 0.0, 0.0])
        obs = np.array([1.0, 0.0, 1.0, 0.0])
        table = PerfMetrics.compute_contingency_table(fcst, obs)
        assert table["hits"] == 1
        assert table["false_alarms"] == 1
        assert table["misses"] == 1
        assert table["correct_negatives"] == 1
        assert table["total"] == 4

    def test_all_hits(self) -> None:
        """
        Confirm that when forecast and observation masks are identical all-ones arrays, all valid points are counted as hits with zero misses and false alarms. This degenerate case represents a perfect forecast for the event category. The correct-negatives count is also zero because every point exceeds the threshold.

                Returns:
            None
        """
        ones = np.array([1.0, 1.0, 1.0])
        table = PerfMetrics.compute_contingency_table(ones, ones)
        assert table["hits"] == 3
        assert table["misses"] == 0
        assert table["false_alarms"] == 0

    def test_all_correct_negatives(self) -> None:
        """
        Verify that when both masks are all-zeros, every point is a correct negative with no hits, misses, or false alarms. This represents the case where no event is predicted or observed at any grid point. The total must still reflect the number of valid input cells to ensure sample-size calculations are correct.
        
        Returns: 
            None
        """
        zeros = np.array([0.0, 0.0, 0.0])
        table = PerfMetrics.compute_contingency_table(zeros, zeros)
        assert table["correct_negatives"] == 3
        assert table["hits"] == 0

    def test_nan_excluded(self) -> None:
        """
        Ensure that grid points where either forecast or observation is NaN are excluded from all contingency table categories. This test places a NaN in the forecast at position 0 and verifies the total drops to 2 rather than 3. NaN exclusion prevents out-of-domain masked cells from being counted as correct negatives or misses.
        """
        fcst = np.array([np.nan, 1.0, 0.0])
        obs = np.array([1.0, 1.0, 0.0])
        table = PerfMetrics.compute_contingency_table(fcst, obs)
        assert table["total"] == 2
        assert table["hits"] == 1
        assert table["correct_negatives"] == 1

    def test_xarray_input(self) -> None:
        """
        Confirm that compute_contingency_table accepts xr.DataArray inputs and produces the same counts as equivalent NumPy arrays. This ensures the method handles the DataArray-to-ndarray extraction transparently so callers do not need to manually extract .values before calling.
        """
        fcst = xr.DataArray([1.0, 0.0, 1.0])
        obs = xr.DataArray([1.0, 0.0, 0.0])
        table = PerfMetrics.compute_contingency_table(fcst, obs)
        assert table["hits"] == 1
        assert table["false_alarms"] == 1
        assert table["correct_negatives"] == 1
        assert table["total"] == 3


# ======================================================================
# Individual contingency-table metrics
# ======================================================================


class TestPOD:
    """Tests for the PerfMetrics.pod static method verifying perfect-detection, partial-detection, and zero-denominator edge cases."""

    def test_perfect(self) -> None:
        """
        Verify that POD is exactly 1.0 when all observed events are correctly forecast, meaning hits equals the total number of observed events and misses is zero. A POD of 1.0 is the theoretical maximum and indicates that no observed events were missed by the forecast.

        Returns:
            None
        """
        table = {"hits": 5, "misses": 0, "false_alarms": 2, "correct_negatives": 3, "total": 10}
        assert PerfMetrics.pod(table) == pytest.approx(1.0)

    def test_partial(self) -> None:
        """
        Confirm that POD correctly reflects the fraction of observed events captured when some are missed. With 3 hits and 2 misses the POD should be 0.6, verifying the basic division logic for a non-degenerate case.
        """
        table = {"hits": 3, "misses": 2, "false_alarms": 1, "correct_negatives": 4, "total": 10}
        assert PerfMetrics.pod(table) == pytest.approx(0.6)

    def test_zero_denominator(self) -> None:
        """
        Ensure POD returns NaN when there are no observed events, making hits + misses equal to zero. This guards against ZeroDivisionError and signals to the caller that the metric is undefined for this case.
        """
        table = {"hits": 0, "misses": 0, "false_alarms": 3, "correct_negatives": 7, "total": 10}
        assert math.isnan(PerfMetrics.pod(table))


class TestFAR:
    """Tests for the PerfMetrics.far static method verifying zero-false-alarm, partial, and zero-denominator cases."""

    def test_no_false_alarms(self) -> None:
        """
        Verify that FAR is exactly 0.0 when the forecast contains no false alarms, meaning every forecast event corresponds to an observed event. A FAR of 0.0 is the ideal value indicating no wasted forecast warnings.
        """
        table = {"hits": 5, "misses": 1, "false_alarms": 0, "correct_negatives": 4, "total": 10}
        assert PerfMetrics.far(table) == pytest.approx(0.0)

    def test_all_false_alarms(self) -> None:
        """
        Confirm that FAR equals 1.0 when all forecast events are false alarms with no hits. This worst-case scenario indicates the forecast has no detection skill and every predicted event is a false alarm.
        """
        table = {"hits": 0, "misses": 3, "false_alarms": 5, "correct_negatives": 2, "total": 10}
        assert PerfMetrics.far(table) == pytest.approx(1.0)

    def test_zero_denominator(self) -> None:
        """
        Ensure FAR returns NaN when there are no forecast events, making hits + false_alarms equal to zero. The metric is undefined without any forecast events to evaluate.
        """
        table = {"hits": 0, "misses": 5, "false_alarms": 0, "correct_negatives": 5, "total": 10}
        assert math.isnan(PerfMetrics.far(table))


class TestCSI:
    """Tests for the PerfMetrics.csi static method covering perfect, partial, and zero-denominator cases, plus the TS alias."""

    def test_perfect(self) -> None:
        """
        Verify that CSI is exactly 1.0 when all observed events are correctly forecast with no false alarms and no misses. This represents a perfect forecast where every event is detected and no spurious events are predicted.
        """
        table = {"hits": 5, "misses": 0, "false_alarms": 0, "correct_negatives": 5, "total": 10}
        assert PerfMetrics.csi(table) == pytest.approx(1.0)

    def test_partial(self) -> None:
        """
        Confirm that CSI returns the correct value for a mixed contingency table with both misses and false alarms. With 4 hits, 2 misses, and 1 false alarm the CSI should be 4/7, verifying the denominator includes all three non-correct-negative categories.
        """
        table = {"hits": 4, "misses": 2, "false_alarms": 1, "correct_negatives": 3, "total": 10}
        assert PerfMetrics.csi(table) == pytest.approx(4.0 / 7.0)

    def test_zero_denominator(self) -> None:
        """
        Ensure CSI returns NaN when hits, misses, and false alarms are all zero, indicating no events in either field. The metric cannot be computed when neither field contains exceedance events.
        """
        table = {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 10, "total": 10}
        assert math.isnan(PerfMetrics.csi(table))

    def test_ts_alias(self) -> None:
        """
        Confirm that PerfMetrics.ts is the same function object as PerfMetrics.csi, verifying the class-level alias is correctly assigned. Threat Score and Critical Success Index are mathematically identical metrics and should share a single implementation.
        """
        assert PerfMetrics.ts is PerfMetrics.csi


class TestFBIAS:
    """Tests for the PerfMetrics.fbias static method verifying unbiased, over-forecasting, under-forecasting, and zero-denominator cases."""

    def test_unbiased(self) -> None:
        """
        Verify that FBIAS is exactly 1.0 when the number of forecast events equals the number of observed events. Perfect frequency bias indicates no systematic over- or under-forecasting of event occurrence regardless of spatial placement accuracy.
        """
        table = {"hits": 4, "misses": 1, "false_alarms": 1, "correct_negatives": 4, "total": 10}
        assert PerfMetrics.fbias(table) == pytest.approx(1.0)

    def test_over_forecast(self) -> None:
        """
        Confirm that FBIAS exceeds 1.0 when more events are forecast than observed, indicating systematic over-forecasting. With 3 hits, 2 misses, and 5 false alarms the FBIAS should be (3+5)/(3+2) = 1.6. An FBIAS above 1.0 is a common diagnostic indicating a wet-biased precipitation forecast that generates more exceedance events than actually observed.

        Returns:
            None
        """
        table = {"hits": 3, "misses": 2, "false_alarms": 5, "correct_negatives": 0, "total": 10}
        assert PerfMetrics.fbias(table) == pytest.approx(1.6)

    def test_under_forecast(self) -> None:
        """
        Verify that FBIAS is less than 1.0 when fewer events are forecast than observed, indicating under-forecasting. With 2 hits, 6 misses, and 0 false alarms the FBIAS should be 2/8 = 0.25. A FBIAS well below 1.0 is characteristic of a dry-biased forecast that predicts far fewer exceedance events than the observation record contains.

        Returns:
            None
        """
        table = {"hits": 2, "misses": 6, "false_alarms": 0, "correct_negatives": 2, "total": 10}
        assert PerfMetrics.fbias(table) == pytest.approx(0.25)

    def test_zero_denominator(self) -> None:
        """
        Ensure FBIAS returns NaN when there are no observed events, making hits + misses equal to zero. The ratio of forecast to observed events is undefined without any observations to compare against. This guards against ZeroDivisionError and signals to the caller that the metric cannot be computed for event-free reference fields.

        Returns:
            None
        """
        table = {"hits": 0, "misses": 0, "false_alarms": 5, "correct_negatives": 5, "total": 10}
        assert math.isnan(PerfMetrics.fbias(table))


class TestETS:
    """Tests for the PerfMetrics.ets static method verifying perfect-forecast, no-skill, and edge cases."""

    def test_perfect(self) -> None:
        """
        Verify that ETS is 1.0 when the forecast perfectly matches the observation with no misses or false alarms. A perfect forecast has hits_random = hits^2/total, so for 5 hits and 10 total, ETS = (5 - 2.5) / (5 - 2.5) = 1.0. This confirms the random-hit correction does not degrade a perfect score and that the formula resolves correctly at its upper boundary.

        Returns:
            None
        """
        table = {"hits": 5, "misses": 0, "false_alarms": 0, "correct_negatives": 5, "total": 10}
        assert PerfMetrics.ets(table) == pytest.approx(1.0)

    def test_no_skill(self) -> None:
        """
        Confirm that ETS produces the expected negative value when the number of hits is below random chance for the given contingency table geometry. This represents a forecast with no skill beyond climatological guessing, where hits_random = (1+4)*(1+4)/10 = 2.5 and ETS = (1-2.5)/(1+4+4-2.5) = -1.5/6.5. A hand-verifiable expected value is computed and compared to guard against errors in the random-hit correction formula.

        Returns:
            None
        """
        # hits_random = (1+4)*(1+4)/10 = 2.5; ETS = (1-2.5)/(1+4+4-2.5) = -1.5/6.5
        table = {"hits": 1, "misses": 4, "false_alarms": 4, "correct_negatives": 1, "total": 10}
        expected = (1 - 2.5) / (1 + 4 + 4 - 2.5)
        assert PerfMetrics.ets(table) == pytest.approx(expected)

    def test_zero_total(self) -> None:
        """
        Ensure ETS returns NaN when the total sample size is zero, making the random-hit correction undefined. This edge case occurs when all grid points are NaN-masked and no valid pairs exist for contingency table construction. Returning NaN rather than raising an exception lets callers detect the undefined case without wrapping every ETS call in a try-except block.

        Returns:
            None
        """
        table = {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0, "total": 0}
        assert math.isnan(PerfMetrics.ets(table))

    def test_zero_denominator(self) -> None:
        """
        Verify that ETS returns NaN when the denominator of the ETS formula is exactly zero, which can occur when hits equals hits_random and there are no misses or false alarms beyond the random expectation. This prevents a ZeroDivisionError from propagating to callers and signals that the score is technically indeterminate for this degenerate table.

        Returns:
            None
        """
        # All correct negatives, no events at all: hits=misses=fa=0, total=10
        # hits_random = 0, denom = 0+0+0-0 = 0
        table = {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 10, "total": 10}
        assert math.isnan(PerfMetrics.ets(table))


# ======================================================================
# Batch metrics integration
# ======================================================================


class TestBatchMetrics:
    """Tests for the extended compute_fss_batch return structure verifying that all contingency-table metrics are present alongside FSS for each (threshold, window) entry."""

    def test_batch_returns_all_metrics(self, pm: PerfMetrics) -> None:
        """
        Verify that compute_fss_batch returns a dictionary where each value is itself a dictionary containing fss, pod, far, csi, fbias, and ets keys. This test uses a perfect-forecast scenario of identical fields to confirm the metric keys are present and that FSS equals 1.0. The remaining metrics are checked for type consistency (float) rather than exact values, since their correct computation is covered by the individual metric-class tests.

        Returns:
            None
        """
        field = xr.DataArray(
            np.random.RandomState(99).rand(20, 20),
            dims=["latitude", "longitude"],
        )
        results = pm.compute_fss_batch(
            field, field,
            thresholds=[90.0],
            window_sizes=[3],
        )
        assert len(results) == 1
        key = (90.0, 3)
        assert key in results
        metrics = results[key]
        for expected_key in ("fss", "pod", "far", "csi", "fbias", "ets"):
            assert expected_key in metrics, f"Missing key: {expected_key}"
            assert isinstance(metrics[expected_key], float)
        assert metrics["fss"] == pytest.approx(1.0, abs=1e-6)


# -----------------------------------------------------------------------
# PerfMetrics gap-closing tests
# -----------------------------------------------------------------------


class TestPerfMetricsNumpyBranches:
    """Cover numpy array branches and save_intermediate paths."""

    @pytest.fixture()
    def pm(self) -> PerfMetrics:
        """
        Construct a PerfMetrics instance backed by default ModvxConfig settings for use in NumPy branch tests. This class-scoped fixture mirrors the module-level pm fixture but is defined locally to avoid fixture-injection ordering issues when tests in this class also require the module-level fixture. Tests in TestPerfMetricsNumpyBranches use raw NumPy arrays to exercise the array-conversion branches that are bypassed when xr.DataArray inputs are always provided.

        Returns:
            PerfMetrics: Fully initialised PerfMetrics object with default configuration.
        """
        return PerfMetrics(ModvxConfig())

    def test_compute_fractional_field_numpy(self, pm: PerfMetrics) -> None:
        """
        Verify that compute_fractional_field accepts a raw NumPy array and returns an xr.DataArray of matching shape. The NumPy input path converts the array to DataArray internally before applying the sliding window. This test confirms the output type and shape are preserved through that conversion to ensure downstream FSS computation can proceed without additional wrapping by the caller.

        Returns:
            None
        """
        arr = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = pm.compute_fractional_field(arr, 3)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (2, 2)

    def test_calculate_fss_numpy(self, pm: PerfMetrics) -> None:
        """
        Verify that calculate_fss accepts raw NumPy arrays directly and returns an FSS value in [0, 1]. This exercises the NumPy input branch of calculate_fss, which should convert arrays to DataArray before computing the fractional fields. The test uses randomly generated independent fields which yield FSS values somewhere in the valid range without requiring a deterministic expected value.

        Returns:
            None
        """
        rng = np.random.default_rng(42)
        fcst = rng.random((20, 20))
        obs = rng.random((20, 20))
        fss = pm.calculate_fss(fcst, obs, 90.0, 3)
        assert 0.0 <= fss <= 1.0

    def test_calculate_fss_save_intermediate(self, pm: PerfMetrics) -> None:
        """
        Verify that calculate_fss calls save_intermediate_binary on the FileManager when save_intermediate is True. The intermediate save path is triggered when both cycle_start and valid_time are supplied alongside the save_intermediate flag. This test mocks the FileManager class so no real disk writes occur, then confirms the save method is called exactly once confirming the FSS pipeline persists the binary mask as expected.

        Returns:
            None
        """
        rng = np.random.default_rng(42)
        fcst = xr.DataArray(rng.random((10, 10)), dims=["latitude", "longitude"])
        obs = xr.DataArray(rng.random((10, 10)), dims=["latitude", "longitude"])

        with patch("modvx.file_manager.FileManager") as mock_fm_cls:
            mock_fm = MagicMock()
            mock_fm_cls.return_value = mock_fm
            pm.calculate_fss(
                fcst, obs, 90.0, 3,
                save_intermediate=True,
                cycle_start=datetime.datetime(2024, 9, 17),
                valid_time=datetime.datetime(2024, 9, 17, 12),
            )
            mock_fm.save_intermediate_binary.assert_called_once()

    def test_compute_fss_batch_save_intermediate(self, pm: PerfMetrics) -> None:
        """
        Verify that compute_fss_batch calls save_intermediate_binary on the FileManager when save_intermediate is True. The batch method should delegate to calculate_fss for each (threshold, window) combination and honour the save_intermediate flag for each call. This test mocks the FileManager and confirms the save method is called exactly once for the single-entry batch confirming proper delegation.

        Returns:
            None
        """
        rng = np.random.default_rng(42)
        fcst = xr.DataArray(rng.random((10, 10)), dims=["latitude", "longitude"])
        obs = xr.DataArray(rng.random((10, 10)), dims=["latitude", "longitude"])

        with patch("modvx.file_manager.FileManager") as mock_fm_cls:
            mock_fm = MagicMock()
            mock_fm_cls.return_value = mock_fm
            results = pm.compute_fss_batch(
                fcst, obs,
                thresholds=[90.0],
                window_sizes=[3],
                save_intermediate=True,
                cycle_start=datetime.datetime(2024, 9, 17),
                valid_time=datetime.datetime(2024, 9, 17, 12),
            )
            assert len(results) == 1
            mock_fm.save_intermediate_binary.assert_called_once()
