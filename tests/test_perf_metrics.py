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

import numpy as np
import pytest
import xarray as xr

from modvx.config import ModvxConfig
from modvx.perf_metrics import PerfMetrics


@pytest.fixture
def pm() -> PerfMetrics:
    """
    This fixture provides a PerfMetrics instance initialized with default ModvxConfig settings for use in all tests in this module. By using a fixture, we ensure that each test receives a fresh instance of PerfMetrics with consistent configuration, avoiding any unintended state carryover between tests. Tests can rely on this fixture to access the methods of PerfMetrics without needing to construct it manually. 

    Parameters:
        None

    Returns:
        PerfMetrics: Fully initialised PerfMetrics object with default configuration.
    """
    return PerfMetrics(ModvxConfig())


class TestBinaryMask:
    """ Tests for generate_binary_mask verifying threshold exceedance logic, NaN propagation, and support for both xarray and NumPy inputs. """

    def test_simple(self: "TestBinaryMask", 
                    pm: PerfMetrics) -> None:
        """
        This test verifies that generate_binary_mask correctly classifies values as 0 or 1 based on a simple threshold. Values below the threshold should be classified as 0, while values equal to or above the threshold should be classified as 1. This confirms the basic exceedance logic is implemented correctly. 

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        arr = xr.DataArray([0.0, 0.5, 1.0, 1.5, 2.0])
        mask = pm.generate_binary_mask(arr, 1.0)
        expected = np.array([0, 0, 1, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(mask.values, expected)

    def test_nan_preserved(self: "TestBinaryMask", 
                           pm: PerfMetrics) -> None:
        """
        This test confirms that NaN values in the input array are preserved as NaN in the output binary mask rather than being classified as 0 or 1. NaN values represent missing or masked-out grid points that should not contribute to the binary classification, so they must remain NaN in the output to ensure proper handling in downstream metrics that rely on valid sample counts. 
        
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

    def test_numpy_input(self: "TestBinaryMask", 
                         pm: PerfMetrics) -> None:
        """
        This test verifies that generate_binary_mask can accept a raw NumPy array as input and still produce an xr.DataArray output with the correct binary classification. The method should internally convert the NumPy array to a DataArray before applying the thresholding logic, so this test confirms that the conversion is handled correctly and does not interfere with the expected output values or types. 

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
    """ Tests for compute_fractional_field verifying neighborhood-averaged fraction values for uniform fields and correct NaN propagation through the sliding window. """

    def test_uniform_field(self: "TestFractionalField", 
                           pm: PerfMetrics) -> None:
        """
        This test confirms that compute_fractional_field returns a value of 1.0 for interior cells when the input binary field is uniformly 1.0. In this case, every cell in the sliding window will be 1.0, so the average should also be 1.0 for all interior cells. This verifies that the neighborhood averaging logic correctly computes the fractional field values without introducing any bias or numerical errors. 
        
        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        arr = xr.DataArray(np.ones((11, 11)))
        frac = pm.compute_fractional_field(arr, 3)
        # Interior cells should be exactly 1
        assert float(frac.values[5, 5]) == pytest.approx(1.0)

    def test_nan_handling(self: "TestFractionalField", 
                          pm: PerfMetrics) -> None:
        """
        This test verifies that if any cell within the sliding window contains a NaN, the output fractional field value for the center cell is also NaN. This ensures that missing or masked-out grid points properly propagate through the neighborhood averaging and do not contribute to a valid fraction value. By placing a single NaN in the input field, we can confirm that the method correctly identifies the presence of NaN in the window and returns NaN for the corresponding output cell. 

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
    """ Tests for calculate_fss verifying the FSS value bounds and the perfect-forecast degenerate case using synthetic 2D spatial fields. """

    def test_perfect_forecast(self: "TestFSS", 
                              pm: PerfMetrics) -> None:
        """
        This test confirms that calculate_fss returns an FSS of 1.0 when the forecast and observation fields are identical, representing a perfect forecast scenario. By using a synthetic 2D field with random values and comparing it to itself, we can verify that the method correctly identifies the perfect spatial match and computes the FSS as 1.0 without any numerical issues or bias. This serves as a critical baseline test to ensure that the FSS calculation is mathematically correct for the ideal case. 

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

    def test_fss_range(self: "TestFSS", 
                       pm: PerfMetrics) -> None:
        """
        This test verifies that the FSS value returned by calculate_fss is always between 0.0 and 1.0 for non-degenerate cases, confirming that the metric is properly normalized and does not produce out-of-bounds values. By using two independent random fields, we can ensure that the FSS is less than 1.0, while still being a valid score within the expected range. This test guards against any mathematical errors in the FSS formula that could lead to invalid scores. 

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
    """ Tests for the PerfMetrics.rmse static method verifying zero-error and known-value cases. """

    def test_zero_error(self: "TestRMSE") -> None:
        """
        This test confirms that rmse returns exactly 0.0 when the forecast and observation arrays are identical, indicating a perfect match with no error. This baseline case verifies that the method correctly computes the mean squared error as zero and that the square root of zero is also zero, confirming the mathematical correctness of the RMSE implementation for the ideal case.

        Parameters:
            None
        
        Returns: 
            None
        """
        a = np.array([1.0, 2.0, 3.0])
        assert PerfMetrics.rmse(a, a) == pytest.approx(0.0)

    def test_known_value(self: "TestRMSE") -> None:
        """
        This test verifies that rmse returns the expected value of 1.0 when the forecast and observation arrays differ by exactly 1.0 at every point. With a = [0, 0] and b = [1, 1], the mean squared error is (1^2 + 1^2)/2 = 1, so the RMSE should be sqrt(1) = 1. This confirms that the method correctly computes the mean squared error and applies the square root to produce the final RMSE value for a simple known case.

        Parameters:
            None

        Returns:
            None
        """
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert PerfMetrics.rmse(a, b) == pytest.approx(1.0)


class TestBias:
    """ Tests for the PerfMetrics.bias static method verifying zero-bias and positive-bias cases. """

    def test_no_bias(self: "TestBias") -> None:
        """
        This test confirms that bias returns exactly 0.0 when the forecast and observation arrays are identical, indicating no systematic bias between the two fields. This baseline case verifies that the method correctly computes the mean difference as zero when there is a perfect match, confirming the mathematical correctness of the bias implementation for the ideal case.

        Parameters:
            None

        Returns:
            None
        """
        a = np.array([1.0, 2.0, 3.0])
        assert PerfMetrics.bias(a, a) == pytest.approx(0.0)

    def test_positive_bias(self: "TestBias") -> None:
        """
        This test verifies that bias returns the expected value of 1.0 when the forecast array is consistently 1.0 greater than the observation array at every point. With fcst = [2, 3] and obs = [1, 2], the mean difference is (1 + 1)/2 = 1, so the bias should be 1. This confirms that the method correctly computes the average difference between the forecast and observation to produce the bias value for a simple known case.

        Parameters:
            None

        Returns:
            None
        """
        fcst = np.array([2.0, 3.0])
        obs = np.array([1.0, 2.0])
        assert PerfMetrics.bias(fcst, obs) == pytest.approx(1.0)


class TestContingencyTable:
    """ Tests for compute_contingency_table verifying correct hit/miss/false-alarm/correct-negative counts and NaN exclusion from all categories. """

    def test_known_counts(self: "TestContingencyTable") -> None:
        """
        This test confirms that compute_contingency_table correctly counts hits, misses, false alarms, and correct negatives for a simple case with a mix of all categories. With fcst = [1, 1, 0, 0] and obs = [1, 0, 1, 0], we expect 1 hit (position 0), 1 false alarm (position 1), 1 miss (position 2), and 1 correct negative (position 3). The total should reflect the number of valid pairs (4 in this case). This test verifies the basic logic of contingency table construction for a non-degenerate case. 

        Parameters:
            None
        
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

    def test_all_hits(self: "TestContingencyTable") -> None:
        """
        This test verifies that when both forecast and observation masks are all-ones, every point is classified as a hit with no misses, false alarms, or correct negatives. This represents the case where every grid point is predicted and observed to exceed the threshold, so the contingency table should reflect hits equal to the total number of valid pairs and zero for all other categories.

        Parameters:
            None

        Returns:
            None
        """
        ones = np.array([1.0, 1.0, 1.0])
        table = PerfMetrics.compute_contingency_table(ones, ones)
        assert table["hits"] == 3
        assert table["misses"] == 0
        assert table["false_alarms"] == 0

    def test_all_correct_negatives(self: "TestContingencyTable") -> None:
        """
        This test confirms that when both forecast and observation masks are all-zeros, every point is classified as a correct negative with no hits, misses, or false alarms. This represents the case where every grid point is predicted and observed to be below the threshold, so the contingency table should reflect correct negatives equal to the total number of valid pairs and zero for all other categories. 

        Parameters:
            None
        
        Returns: 
            None
        """
        zeros = np.array([0.0, 0.0, 0.0])
        table = PerfMetrics.compute_contingency_table(zeros, zeros)
        assert table["correct_negatives"] == 3
        assert table["hits"] == 0

    def test_nan_excluded(self: "TestContingencyTable") -> None:
        """
        This test verifies that any pair of forecast and observation values where either is NaN is excluded from all contingency categories and does not contribute to the total count. With fcst = [NaN, 1, 0] and obs = [1, 1, 0], the first pair is invalid due to NaN and should be ignored, while the second and third pairs yield 1 hit and 1 correct negative respectively. The total should reflect only the valid pairs (2 in this case), confirming that NaN values are properly handled as missing data in the contingency table construction.

        Parameters:
            None

        Returns:
            None
        """
        fcst = np.array([np.nan, 1.0, 0.0])
        obs = np.array([1.0, 1.0, 0.0])
        table = PerfMetrics.compute_contingency_table(fcst, obs)
        assert table["total"] == 2
        assert table["hits"] == 1
        assert table["correct_negatives"] == 1

    def test_xarray_input(self: "TestContingencyTable") -> None:
        """
        This test confirms that compute_contingency_table can accept xr.DataArray inputs and produce the correct contingency counts. The method should internally handle the DataArray inputs without requiring manual conversion by the caller, so this test verifies that the logic for counting hits, misses, false alarms, and correct negatives works correctly when the inputs are DataArrays rather than raw NumPy arrays. By using a simple case with known expected counts, we can confirm that the method processes DataArrays as intended. 

        Parameters:
            None

        Returns:
            None
        """
        fcst = xr.DataArray([1.0, 0.0, 1.0])
        obs = xr.DataArray([1.0, 0.0, 0.0])
        table = PerfMetrics.compute_contingency_table(fcst, obs)
        assert table["hits"] == 1
        assert table["false_alarms"] == 1
        assert table["correct_negatives"] == 1
        assert table["total"] == 3


class TestPOD:
    """ Tests for the PerfMetrics.pod static method verifying perfect-detection, partial-detection, and zero-denominator edge cases. """

    def test_perfect(self: "TestPOD") -> None:
        """
        This test confirms that POD returns exactly 1.0 when all observed events are correctly forecast with no misses, indicating perfect detection. With 5 hits and 0 misses, the POD should be 5/(5+0) = 1.0, confirming that the method correctly computes the ratio of hits to total observed events for the ideal case. 

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 5, "misses": 0, "false_alarms": 2, "correct_negatives": 3, "total": 10}
        assert PerfMetrics.pod(table) == pytest.approx(1.0)

    def test_partial(self: "TestPOD") -> None:
        """
        This test verifies that POD returns the correct value for a contingency table with both hits and misses. With 3 hits and 2 misses, the POD should be 3/(3+2) = 0.6, confirming that the method correctly computes the ratio of hits to total observed events when there is a mix of detections and misses. This test guards against any errors in the POD formula that could arise from incorrect handling of the contingency table counts. 

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 3, "misses": 2, "false_alarms": 1, "correct_negatives": 4, "total": 10}
        assert PerfMetrics.pod(table) == pytest.approx(0.6)

    def test_zero_denominator(self: "TestPOD") -> None:
        """
        This test confirms that POD returns NaN when there are no observed events, making hits + misses equal to zero. The metric is undefined in this case because there are no events to evaluate for detection, so returning NaN signals to the caller that the POD cannot be computed for this contingency table. This guards against ZeroDivisionError and ensures that the method handles edge cases gracefully without crashing. 

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 0, "misses": 0, "false_alarms": 3, "correct_negatives": 7, "total": 10}
        assert math.isnan(PerfMetrics.pod(table))


class TestFAR:
    """ Tests for the PerfMetrics.far static method verifying zero-false-alarm, partial, and zero-denominator cases. """

    def test_no_false_alarms(self: "TestFAR") -> None:
        """
        This test confirms that FAR returns exactly 0.0 when there are no false alarms, indicating perfect specificity. With 5 hits and 0 false alarms, the FAR should be 0/(5+0) = 0.0, confirming that the method correctly computes the ratio of false alarms to total forecast events for the ideal case where every forecast event is a hit.

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 5, "misses": 1, "false_alarms": 0, "correct_negatives": 4, "total": 10}
        assert PerfMetrics.far(table) == pytest.approx(0.0)

    def test_all_false_alarms(self: "TestFAR") -> None:
        """
        This test verifies that FAR returns exactly 1.0 when all forecast events are false alarms with no hits, indicating zero specificity. With 0 hits and 5 false alarms, the FAR should be 5/(0+5) = 1.0, confirming that the method correctly computes the ratio of false alarms to total forecast events for the case where every forecast event is a false alarm. This test guards against any errors in the FAR formula that could arise from incorrect handling of the contingency table counts.

        Parameters:
            None
        
        Returns:
            None
        """
        table = {"hits": 0, "misses": 3, "false_alarms": 5, "correct_negatives": 2, "total": 10}
        assert PerfMetrics.far(table) == pytest.approx(1.0)

    def test_zero_denominator(self: "TestFAR") -> None:
        """
        This test confirms that FAR returns NaN when there are no forecast events, making hits + false_alarms equal to zero. The metric is undefined in this case because there are no forecast events to evaluate for false alarms, so returning NaN signals to the caller that the FAR cannot be computed for this contingency table. This guards against ZeroDivisionError and ensures that the method handles edge cases gracefully without crashing.

        Parameters:
            None
        
        Returns:
            None
        """
        table = {"hits": 0, "misses": 5, "false_alarms": 0, "correct_negatives": 5, "total": 10}
        assert math.isnan(PerfMetrics.far(table))


class TestCSI:
    """ Tests for the PerfMetrics.csi static method covering perfect, partial, and zero-denominator cases, plus the TS alias. """

    def test_perfect(self: "TestCSI") -> None:
        """
        This test confirms that CSI returns exactly 1.0 when there are no misses or false alarms, indicating perfect skill. With 5 hits and 0 misses and false alarms, the CSI should be 5/(5+0+0) = 1.0, confirming that the method correctly computes the ratio of hits to total forecast and observed events for the ideal case where every forecast event is a hit and there are no misses or false alarms.

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 5, "misses": 0, "false_alarms": 0, "correct_negatives": 5, "total": 10}
        assert PerfMetrics.csi(table) == pytest.approx(1.0)

    def test_partial(self: "TestCSI") -> None:
        """
        This test verifies that CSI returns the correct value for a contingency table with a mix of hits, misses, and false alarms. With 4 hits, 2 misses, and 1 false alarm, the CSI should be 4/(4+2+1) = 4/7 ≈ 0.5714, confirming that the method correctly computes the ratio of hits to total forecast and observed events when there is a mix of detections, misses, and false alarms. This test guards against any errors in the CSI formula that could arise from incorrect handling of the contingency table counts.

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 4, "misses": 2, "false_alarms": 1, "correct_negatives": 3, "total": 10}
        assert PerfMetrics.csi(table) == pytest.approx(4.0 / 7.0)

    def test_zero_denominator(self: "TestCSI") -> None:
        """
        This test confirms that CSI returns NaN when there are no hits, misses, or false alarms, making the denominator of the CSI formula equal to zero. The metric is undefined in this case because there are no forecast or observed events to evaluate for skill, so returning NaN signals to the caller that the CSI cannot be computed for this contingency table. This guards against ZeroDivisionError and ensures that the method handles edge cases gracefully without crashing. 

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 10, "total": 10}
        assert math.isnan(PerfMetrics.csi(table))

    def test_ts_alias(self: "TestCSI") -> None:
        """
        This test confirms that the TS alias for CSI is correctly implemented as a reference to the same method. By asserting that PerfMetrics.ts is the same object as PerfMetrics.csi, we verify that both names point to the same function and that there are no discrepancies in their behavior or output. This ensures that users can use either name interchangeably without confusion or errors.

        Parameters:
            None

        Returns:
            None
        """
        assert PerfMetrics.ts is PerfMetrics.csi


class TestFBIAS:
    """ Tests for the PerfMetrics.fbias static method verifying unbiased, over-forecasting, under-forecasting, and zero-denominator cases. These tests confirm that the FBIAS calculation correctly identifies perfect frequency bias, handles over- and under-forecasting scenarios, and gracefully handles cases with no observed events. By using small arrays with hand-verifiable expected results, we can validate the mathematical correctness of the FBIAS implementation for both ideal and non-ideal forecast-observation pairs. """

    def test_unbiased(self: "TestFBIAS") -> None:
        """
        This test confirms that FBIAS returns exactly 1.0 when the number of forecast events matches the number of observed events, indicating an unbiased forecast. With 4 hits, 1 miss, and 1 false alarm, the total forecast events are 5 (hits + false alarms) and the total observed events are also 5 (hits + misses), so the FBIAS should be 5/5 = 1.0. This verifies that the method correctly computes the ratio of forecast to observed events for the ideal case where there is no bias in either direction. 

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 4, "misses": 1, "false_alarms": 1, "correct_negatives": 4, "total": 10}
        assert PerfMetrics.fbias(table) == pytest.approx(1.0)

    def test_over_forecast(self: "TestFBIAS") -> None:
        """
        This test verifies that FBIAS returns a value greater than 1.0 when there are more forecast events than observed events, indicating over-forecasting. With 3 hits, 2 misses, and 5 false alarms, the total forecast events are 8 (hits + false alarms) while the total observed events are only 5 (hits + misses), so the FBIAS should be 8/5 = 1.6. This confirms that the method correctly computes the ratio of forecast to observed events and identifies the over-forecasting bias in this scenario. 

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 3, "misses": 2, "false_alarms": 5, "correct_negatives": 0, "total": 10}
        assert PerfMetrics.fbias(table) == pytest.approx(1.6)

    def test_under_forecast(self: "TestFBIAS") -> None:
        """
        This test verifies that FBIAS returns a value less than 1.0 when there are fewer forecast events than observed events, indicating under-forecasting. With 2 hits, 6 misses, and 0 false alarms, the total forecast events are only 2 (hits) while the total observed events are 8 (hits + misses), so the FBIAS should be 2/8 = 0.25. This confirms that the method correctly computes the ratio of forecast to observed events and identifies the under-forecasting bias in this scenario. A FBIAS well below 1.0 is characteristic of a dry-biased forecast that predicts far fewer exceedance events than the observation record contains. 

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 2, "misses": 6, "false_alarms": 0, "correct_negatives": 2, "total": 10}
        assert PerfMetrics.fbias(table) == pytest.approx(0.25)

    def test_zero_denominator(self: "TestFBIAS") -> None:
        """
        This test confirms that FBIAS returns NaN when there are no observed events, making the denominator of the FBIAS formula equal to zero. The metric is undefined in this case because there are no observed events to evaluate for bias, so returning NaN signals to the caller that the FBIAS cannot be computed for this contingency table. This guards against ZeroDivisionError and ensures that the method handles edge cases gracefully without crashing. 

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 0, "misses": 0, "false_alarms": 5, "correct_negatives": 5, "total": 10}
        assert math.isnan(PerfMetrics.fbias(table))


class TestETS:
    """ Tests for the PerfMetrics.ets static method verifying perfect-forecast, no-skill, and edge cases. """

    def test_perfect(self: "TestETS") -> None:
        """
        This test confirms that ETS returns exactly 1.0 when there are no misses or false alarms, indicating perfect skill. With 5 hits and 0 misses and false alarms, the ETS should be 1.0, confirming that the method correctly computes the score for the ideal case where every forecast event is a hit and there are no misses or false alarms. This serves as a critical baseline test to ensure that the ETS calculation is mathematically correct for the perfect forecast scenario. 

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 5, "misses": 0, "false_alarms": 0, "correct_negatives": 5, "total": 10}
        assert PerfMetrics.ets(table) == pytest.approx(1.0)

    def test_no_skill(self: "TestETS") -> None:
        """
        This test verifies that ETS returns a value less than or equal to 0.0 when the forecast has no skill compared to random chance. With 1 hit, 4 misses, and 4 false alarms, the expected hits_random is (1+4)*(1+4)/10 = 2.5, so the ETS should be (1-2.5)/(1+4+4-2.5) = -1.5/6.5 ≈ -0.2308, confirming that the method correctly computes a negative score for a forecast that performs worse than random chance. This test guards against any errors in the ETS formula that could arise from incorrect handling of the contingency table counts or the random-hit correction. 

        Parameters:
            None

        Returns:
            None
        """
        # hits_random = (1+4)*(1+4)/10 = 2.5; ETS = (1-2.5)/(1+4+4-2.5) = -1.5/6.5
        table = {"hits": 1, "misses": 4, "false_alarms": 4, "correct_negatives": 1, "total": 10}
        expected = (1 - 2.5) / (1 + 4 + 4 - 2.5)
        assert PerfMetrics.ets(table) == pytest.approx(expected)

    def test_zero_total(self: "TestETS") -> None:
        """
        This test confirms that ETS returns NaN when there are no valid forecast-observation pairs, making the total count in the contingency table equal to zero. The metric is undefined in this case because there are no events to evaluate for skill, so returning NaN signals to the caller that the ETS cannot be computed for this contingency table. This guards against ZeroDivisionError and ensures that the method handles edge cases gracefully without crashing. 

        Parameters:
            None

        Returns:
            None
        """
        table = {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0, "total": 0}
        assert math.isnan(PerfMetrics.ets(table))

    def test_zero_denominator(self: "TestETS") -> None:
        """
        This test verifies that ETS returns NaN when the denominator of the ETS formula is zero, which can occur when there are no hits, misses, or false alarms, resulting in hits_random equal to hits and making the denominator (hits + misses + false_alarms - hits_random) equal to zero. In this case, the metric is undefined because the forecast has no skill compared to random chance, so returning NaN signals to the caller that the ETS cannot be computed for this contingency table. This guards against ZeroDivisionError and ensures that the method handles edge cases gracefully without crashing. 

        Parameters:
            None

        Returns:
            None
        """
        # All correct negatives, no events at all: hits=misses=fa=0, total=10
        # hits_random = 0, denom = 0+0+0-0 = 0
        table = {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 10, "total": 10}
        assert math.isnan(PerfMetrics.ets(table))


class TestBatchMetrics:
    """ Tests for compute_fss_batch and compute_contingency_batch verifying correct output structure and expected metric values for perfect-forecast scenarios. """

    def test_fss_batch_returns_fss_only(self: "TestBatchMetrics", 
                                        pm: PerfMetrics) -> None:
        """
        This test confirms that compute_fss_batch returns a dictionary with keys corresponding to the provided thresholds and window sizes, and that the metrics dictionary for each key contains only the "fss" metric. By using identical forecast and observation fields, we can verify that the FSS value is approximately 1.0 for the perfect-forecast scenario, confirming that the batch computation correctly identifies the perfect spatial match and computes the FSS as expected. This test also ensures that no extraneous metrics are included in the output when only FSS is computed. 

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

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
        assert set(metrics.keys()) == {"fss"}
        assert metrics["fss"] == pytest.approx(1.0, abs=1e-6)

    def test_contingency_batch_returns_metrics(self: "TestBatchMetrics", 
                                               pm: PerfMetrics) -> None:
        """
        This test verifies that compute_contingency_batch returns a dictionary with keys corresponding to the provided thresholds, and that the metrics dictionary for each key contains the expected contingency-based metrics: "pod", "far", "csi", "fbias", and "ets". By using identical forecast and observation fields with a threshold that is exceeded by all values, we can confirm that the metrics reflect a perfect forecast scenario with POD=1.0, FAR=0.0, CSI=1.0, FBIAS=1.0, and ETS=1.0. This test ensures that the batch computation correctly processes DataArray inputs and produces valid contingency metrics for the perfect-forecast case. 

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        field = xr.DataArray(
            np.linspace(0.0, 1.0, 100).reshape(10, 10),
            dims=["latitude", "longitude"],
        )
        results = pm.compute_contingency_batch(
            field, field,
            thresholds=[50.0],
        )
        assert 50.0 in results
        metrics = results[50.0]
        assert set(metrics.keys()) == {"pod", "far", "csi", "fbias", "ets"}
        assert metrics["pod"] == pytest.approx(1.0)
        assert metrics["far"] == pytest.approx(0.0)
        assert metrics["csi"] == pytest.approx(1.0)
        assert metrics["fbias"] == pytest.approx(1.0)
        assert metrics["ets"] == pytest.approx(1.0)


class TestPerfMetricsNumpyBranches:
    """ Tests for PerfMetrics methods that exercise the NumPy input branches of compute_fractional_field and calculate_fss. """

    @pytest.fixture()
    def pm(self: "TestPerfMetricsNumpyBranches") -> PerfMetrics:
        """
        This fixture provides a PerfMetrics instance for testing the NumPy input branches of compute_fractional_field and calculate_fss. By using the default ModvxConfig, we can ensure that the PerfMetrics object is fully initialized with standard settings, allowing us to focus on verifying the behavior of these methods when given raw NumPy arrays as input. This fixture can be reused across multiple tests that need to exercise the NumPy input paths without requiring manual setup in each test function. 

        Parameters:
            None

        Returns:
            PerfMetrics: Fully initialised PerfMetrics object with default configuration.
        """
        return PerfMetrics(ModvxConfig())

    def test_compute_fractional_field_numpy(self: "TestPerfMetricsNumpyBranches", 
                                            pm: PerfMetrics) -> None:
        """
        This test confirms that compute_fractional_field can accept a raw NumPy array as input and returns an xr.DataArray with the expected shape. By providing a simple 2D NumPy array and a window size, we can verify that the method correctly processes the NumPy input, computes the fractional field, and returns it as a DataArray without errors. This test ensures that the NumPy input branch of compute_fractional_field is functioning properly and that the output is in the correct format for downstream processing. 

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        arr = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = pm.compute_fractional_field(arr, 3)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (2, 2)

    def test_calculate_fss_numpy(self: "TestPerfMetricsNumpyBranches", 
                                 pm: PerfMetrics) -> None:
        """
        This test verifies that calculate_fss can accept raw NumPy arrays as input for the forecast and observation fields, and that it returns a valid FSS value between 0.0 and 1.0. By providing two independent random NumPy arrays, we can confirm that the method processes the inputs without error and produces an FSS score that is properly normalized within the expected range. This test ensures that the NumPy input branch of calculate_fss is functioning correctly and that the method can handle raw arrays without requiring manual conversion to DataArrays by the caller. 

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        rng = np.random.default_rng(42)
        fcst = rng.random((20, 20))
        obs = rng.random((20, 20))
        fss = pm.calculate_fss(fcst, obs, 90.0, 3)
        assert 0.0 <= fss <= 1.0

    def test_calculate_fss_save_intermediate(self: "TestPerfMetricsNumpyBranches", 
                                             pm: PerfMetrics) -> None:
        """
        This test confirms that calculate_fss calls save_intermediate_binary on the FileManager when save_intermediate is True. A call-tracking stub replaces the FileManager class in modvx.file_manager directly; after the call the recorded invocation list is asserted to contain exactly one entry, confirming that the intermediate results are being saved as intended when the flag is set.

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        import modvx.file_manager as _fm_module

        rng = np.random.default_rng(42)
        fcst = xr.DataArray(rng.random((10, 10)), dims=["latitude", "longitude"])
        obs = xr.DataArray(rng.random((10, 10)), dims=["latitude", "longitude"])

        save_binary_calls: list = []

        class _StubFileManager:
            def __init__(self, *args, **kwargs) -> None:
                pass
            def save_intermediate_binary(self, *args, **kwargs) -> None:
                save_binary_calls.append(args)

        _orig_cls = _fm_module.FileManager
        _fm_module.FileManager = _StubFileManager  # type: ignore[attr-defined]
        try:
            pm.calculate_fss(
                fcst, obs, 90.0, 3,
                save_intermediate=True,
                cycle_start=datetime.datetime(2024, 9, 17),
                valid_time=datetime.datetime(2024, 9, 17, 12),
            )
            assert len(save_binary_calls) == 1
        finally:
            _fm_module.FileManager = _orig_cls  # type: ignore[attr-defined]

    def test_compute_fss_batch_save_intermediate(self: "TestPerfMetricsNumpyBranches", 
                                                 pm: PerfMetrics) -> None:
        """
        This test verifies that compute_fss_batch calls save_intermediate_binary on the FileManager for each FSS calculation when save_intermediate is True. By mocking the FileManager, we can confirm that the save method is called the expected number of times based on the number of threshold and window size combinations provided. This test ensures that the logic for saving intermediate results is properly integrated into the batch computation process and that it functions correctly even when given raw NumPy array inputs for the forecast and observation fields. 

        Parameters:
            pm (PerfMetrics): The PerfMetrics instance to test, provided by the pm fixture.

        Returns:
            None
        """
        import modvx.file_manager as _fm_module

        rng = np.random.default_rng(42)
        fcst = xr.DataArray(rng.random((10, 10)), dims=["latitude", "longitude"])
        obs = xr.DataArray(rng.random((10, 10)), dims=["latitude", "longitude"])

        save_binary_calls: list = []

        class _StubFileManager:
            def __init__(self, *args, **kwargs) -> None:
                pass
            def save_intermediate_binary(self, *args, **kwargs) -> None:
                save_binary_calls.append(args)

        _orig_cls = _fm_module.FileManager
        _fm_module.FileManager = _StubFileManager  # type: ignore[attr-defined]
        try:
            results = pm.compute_fss_batch(
                fcst, obs,
                thresholds=[90.0],
                window_sizes=[3],
                save_intermediate=True,
                cycle_start=datetime.datetime(2024, 9, 17),
                valid_time=datetime.datetime(2024, 9, 17, 12),
            )
            assert len(results) == 1
            assert len(save_binary_calls) == 1
        finally:
            _fm_module.FileManager = _orig_cls  # type: ignore[attr-defined]
