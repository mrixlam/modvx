#!/usr/bin/env python3

"""
Tests for modvx.parallel — backend selection, grouping, and execution.

This module contains unit and integration tests for the ParallelProcessor class and its associated logic. Tests cover backend selection based on configuration, correct grouping of work-units by cycle and region, and successful execution of parallel tasks with mock processing functions. The goal is to ensure that the parallel processing framework correctly orchestrates the distribution of work across available resources while adhering to the smart grouping strategy that minimizes redundant data loading.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from __future__ import annotations

import datetime
import os
import types
from unittest.mock import MagicMock, patch

from modvx.parallel import (
    ParallelProcessor,
    _initialize_multiprocessing_worker,
    _multiprocessing_worker_execute_units,
)


# -----------------------------------------------------------------------
# Helper to build work-unit dicts
# -----------------------------------------------------------------------

def _make_unit(cycle: str, region: str) -> dict:
    """
    Build a minimal work-unit dictionary from a cycle string and region name. The returned dictionary contains cycle_start as a parsed datetime and region_name as a string, matching the structure expected by ParallelProcessor and TaskManager. This helper avoids repeating the datetime.strptime pattern across every test that needs synthetic work units.

    Parameters:
        cycle (str): Forecast cycle string in 'YYYYMMDDhh' format.
        region (str): Verification region name (e.g., 'GLOBAL', 'TROPICS').

    Returns:
        dict: Work-unit dictionary with 'cycle_start' (datetime) and 'region_name' (str) keys.
    """
    return {
        "cycle_start": datetime.datetime.strptime(cycle, "%Y%m%d%H"),
        "region_name": region,
    }


# -----------------------------------------------------------------------
# Backend resolution
# -----------------------------------------------------------------------

class TestResolveBackend:
    """Tests for _resolve_backend static method."""

    def test_explicit_serial(self) -> None:
        """
        Verify that _resolve_backend returns 'serial' when the backend argument is explicitly 'serial'. Explicit backend specification should always bypass auto-detection and return the requested string unchanged. This ensures user-requested serial mode is never overridden by environment variable detection.

        Returns:
            None
        """
        assert ParallelProcessor._resolve_backend("serial") == "serial"

    def test_explicit_multiprocessing(self) -> None:
        """
        Verify that _resolve_backend returns 'multiprocessing' when explicitly requested. Like other explicit modes, the multiprocessing backend should bypass MPI environment variable checks and return the string as-is. This test guards against accidental downgrade to serial when the multiprocessing flag is set.

        Returns:
            None
        """
        assert ParallelProcessor._resolve_backend("multiprocessing") == "multiprocessing"

    def test_explicit_mpi(self) -> None:
        """
        Verify that _resolve_backend returns 'mpi' when explicitly requested by the caller. Explicit MPI mode should be returned without checking whether mpi4py is importable, since the caller has declared the intent. Actual availability is validated later when the ParallelProcessor is instantiated and _ensure_mpi is called.

        Returns:
            None
        """
        assert ParallelProcessor._resolve_backend("mpi") == "mpi"

    def test_auto_defaults_serial(self) -> None:
        """
        Verify that 'auto' backend resolves to 'serial' when no MPI environment variables are set. In the absence of MPI launcher environment variables such as OMPI_COMM_WORLD_SIZE, the auto-detection logic should choose the safe serial fallback. This is the expected default on developer workstations and in CI where no MPI launcher is present.

        Returns:
            None
        """
        with patch.dict("os.environ", {}, clear=True):
            result = ParallelProcessor._resolve_backend("auto")
            assert result == "serial"


# -----------------------------------------------------------------------
# Grouping helpers
# -----------------------------------------------------------------------

class TestGrouping:
    """Tests for _group_key, _build_groups, and _assign_groups_round_robin."""

    def test_group_key(self) -> None:
        """
        Verify that _group_key returns the cycle start string, used to co-locate
        work units that share the same forecast cycle on the same parallel worker.

        Returns:
            None
        """
        unit = _make_unit("2024091700", "GLOBAL")
        key = ParallelProcessor._unit_cycle_key(unit)
        assert key == "2024091700"

    def test_build_groups(self) -> None:
        """
        Verify that _build_groups groups work units by cycle, so all regions for the
        same cycle are in one group.  This maximises forecast data reuse in the cache.

        Returns:
            None
        """
        units = [
            _make_unit("2024091700", "GLOBAL"),
            _make_unit("2024091700", "TROPICS"),
            _make_unit("2024091800", "TROPICS"),
        ]
        groups = ParallelProcessor._group_units_by_cycle(units)
        assert len(groups) == 2
        assert len(groups["2024091700"]) == 2
        assert len(groups["2024091800"]) == 1

    def test_round_robin_assignment(self) -> None:
        """
        Verify that _assign_groups_round_robin distributes all work units across the requested number of workers. This test creates three single-cycle groups and assigns them to two workers, then checks that the total across all worker assignments equals three. Round-robin ensures load is spread as evenly as possible when the number of groups is not a multiple of the worker count.

        Returns:
            None
        """
        units = [
            _make_unit("2024091700", "GLOBAL"),
            _make_unit("2024091800", "GLOBAL"),
            _make_unit("2024091900", "GLOBAL"),
        ]
        fn = MagicMock()
        pp = ParallelProcessor(fn, backend="serial")
        assignment = pp._assign_groups_to_workers_round_robin(units, 2)
        # 3 groups → worker 0 gets 2, worker 1 gets 1 (round robin)
        total = sum(len(v) for v in assignment.values())
        assert total == 3

    def test_same_group_same_worker(self) -> None:
        """
        Verify that work units sharing the same cycle are assigned to the same worker,
        so the in-memory forecast cache is reused across regions.

        Returns:
            None
        """
        units = [
            _make_unit("2024091700", "GLOBAL"),
            _make_unit("2024091700", "TROPICS"),
            _make_unit("2024091800", "NAMERICA"),
        ]
        fn = MagicMock()
        pp = ParallelProcessor(fn, backend="serial")
        assignment = pp._assign_groups_to_workers_round_robin(units, 2)
        # Both cycle-1700 units (GLOBAL + TROPICS) should be in the same worker
        for worker_units in assignment.values():
            cycles = {u["cycle_start"].strftime("%Y%m%d%H") for u in worker_units}
            assert len(cycles) == 1, (
                "Units from different cycles should not share a worker bucket"
            )


# -----------------------------------------------------------------------
# Serial backend
# -----------------------------------------------------------------------

class TestSerialBackend:
    """Tests for serial execution path."""

    def test_serial_executes_all_units(self) -> None:
        """
        Verify that the serial backend calls the execution function once for every work unit. This test mocks the function and confirms it is called exactly twice when two units are passed. The serial path must process every unit without skipping, batching, or raising exceptions for this simple case.

        Returns:
            None
        """
        fn = MagicMock()
        pp = ParallelProcessor(fn, backend="serial")
        units = [
            _make_unit("2024091700", "GLOBAL"),
            _make_unit("2024091800", "GLOBAL"),
        ]
        pp.run(units)
        assert fn.call_count == 2

    def test_serial_order_preserved(self) -> None:
        """
        Verify that the serial backend processes work units in the order they appear in the input list. Deterministic ordering in serial mode is important for reproducible debugging and for cases where intermediate files from one unit are read by a later unit. This test records call order via a tracking function and asserts GLOBAL precedes TROPICS.

        Returns:
            None
        """
        calls = []
        def track_fn(unit):
            calls.append(unit["region_name"])

        pp = ParallelProcessor(track_fn, backend="serial")
        units = [
            _make_unit("2024091700", "GLOBAL"),
            _make_unit("2024091800", "TROPICS"),
        ]
        pp.run(units)
        assert calls == ["GLOBAL", "TROPICS"]


# -----------------------------------------------------------------------
# Multiprocessing worker functions
# -----------------------------------------------------------------------

class TestMpWorkerFunctions:
    """Tests for _mp_init_worker and _mp_worker module-level functions."""

    def test_init_worker_sets_global(self) -> None:
        """
        Verify that _mp_init_worker stores the execution function in the module-level _MP_EXECUTE_FN global. Worker processes in a multiprocessing pool need access to the shared function without pickling it on every call, so it is stored as a module global during pool initialisation. This test confirms the global is set to the provided mock after calling _mp_init_worker.

        Returns:
            None
        """
        import modvx.parallel as par
        fn = MagicMock()
        _initialize_multiprocessing_worker(fn)
        assert par._MP_EXECUTE_FN is fn

    def test_mp_worker_executes(self) -> None:
        """
        Verify that _mp_worker calls the initialised function for each unit in its assigned list and returns the count. This test calls _mp_init_worker first to set the worker function, then passes a one-unit list to _mp_worker and asserts the return value is 1 and the mock was called once. The count return value is used by the parent process to tally completed units.

        Returns:
            None
        """
        fn = MagicMock()
        _initialize_multiprocessing_worker(fn)
        units = [_make_unit("2024091700", "GLOBAL")]
        count = _multiprocessing_worker_execute_units(units)
        assert count == 1
        fn.assert_called_once()


# -----------------------------------------------------------------------
# Multiprocessing backend (with mocked Pool)
# -----------------------------------------------------------------------

class TestMultiprocessingBackend:
    """Tests for multiprocessing backend using mocked Pool."""

    def test_pool_called(self) -> None:
        """
        Verify that the multiprocessing backend creates a Pool and calls its map method with the worker function. The Pool is mocked to avoid spawning real processes during unit testing. This test confirms that the correct Pool context manager interface is used and that map is invoked exactly once with the worker and the grouped work-unit lists.

        Returns:
            None
        """
        fn = MagicMock()
        pp = ParallelProcessor(fn, backend="multiprocessing", nprocs=2)
        units = [
            _make_unit("2024091700", "GLOBAL"),
            _make_unit("2024091800", "GLOBAL"),
        ]
        with patch("modvx.parallel.mp.Pool") as mock_pool:
            mock_ctx = MagicMock()
            mock_pool.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.map.return_value = [1, 1]

            pp.run(units)
            mock_ctx.map.assert_called_once()


# -----------------------------------------------------------------------
# Properties
# -----------------------------------------------------------------------

class TestProperties:
    """Tests for ParallelProcessor properties."""

    def test_is_root_serial(self) -> None:
        """
        Verify that is_root returns True for a serial-backend ParallelProcessor. In serial mode there is only one process and it is always the root, so is_root must return True unconditionally. This property is used by callers to gate logging and output operations that should only execute on the controlling process.

        Returns:
            None
        """
        fn = MagicMock()
        pp = ParallelProcessor(fn, backend="serial")
        assert pp.is_root is True

    def test_rank_serial(self) -> None:
        """
        Verify that rank is 0 and size is 1 for a serial-backend ParallelProcessor. Serial execution is equivalent to a single-rank MPI job, so rank must be 0 and size 1 to keep calling code consistent whether MPI or serial is used. This test checks both properties in one assertion pair.

        Returns:
            None
        """
        fn = MagicMock()
        pp = ParallelProcessor(fn, backend="serial")
        assert pp.rank == 0
        assert pp.size == 1


# -----------------------------------------------------------------------
# _ensure_mpi and auto-detection branches
# -----------------------------------------------------------------------


class TestEnsureMpiBranches:
    """Cover _ensure_mpi success/failure paths and auto-detection of MPI."""

    def test_resolve_backend_auto_with_mpi_env(self) -> None:
        """
        Verify that 'auto' resolves to 'serial' when mpi4py is not available even if MPI env vars are set. The presence of OMPI_COMM_WORLD_SIZE signals that the process was launched by an MPI runner, but if mpi4py cannot be imported the backend must fall back gracefully to serial. This test mocks _ensure_mpi to return False, simulating a missing mpi4py installation.

        Returns:
            None
        """
        import modvx.parallel as par

        old = par._HAS_MPI
        par._HAS_MPI = None
        with patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "4"}):
            with patch("modvx.parallel._ensure_mpi", return_value=False):
                result = par.ParallelProcessor._resolve_backend("auto")
                assert result == "serial"
        par._HAS_MPI = old

    def test_ensure_mpi_import_failure(self) -> None:
        """
        Verify that _ensure_mpi returns False and sets _HAS_MPI to False when mpi4py is not installed. The function probes for mpi4py at runtime and must fail gracefully rather than raising an ImportError. After a failed attempt, _HAS_MPI is cached as False so subsequent calls skip the import check entirely.

        Returns:
            None
        """
        import modvx.parallel as par

        old_has = par._HAS_MPI
        old_mpi = par.MPI
        par._HAS_MPI = None
        par.MPI = None
        with patch.dict("sys.modules", {"mpi4py": None, "mpi4py.MPI": None}):
            result = par._ensure_mpi()
            assert result is False
            assert par._HAS_MPI is False
        par._HAS_MPI = old_has
        par.MPI = old_mpi

    def test_ensure_mpi_success(self) -> None:
        """
        Verify that _ensure_mpi returns True and sets _HAS_MPI to True when mpi4py imports successfully. This test replaces the sys.modules entries with fake mpi4py objects that expose the COMM_WORLD interface. After a successful import, subsequent calls to _ensure_mpi should return immediately without re-importing.

        Returns:
            None
        """
        import modvx.parallel as par

        old_has = par._HAS_MPI
        old_mpi = par.MPI
        par._HAS_MPI = None

        fake_mpi_mod = types.ModuleType("mpi4py.MPI")
        setattr(fake_mpi_mod, "COMM_WORLD", MagicMock())
        fake_mpi_mod.COMM_WORLD.Get_size.return_value = 4 
        fake_mpi_mod.COMM_WORLD.Get_rank.return_value = 0 

        fake_mpi4py = types.ModuleType("mpi4py")
        setattr(fake_mpi4py, "MPI", fake_mpi_mod)

        with patch.dict("sys.modules", {
            "mpi4py": fake_mpi4py,
            "mpi4py.MPI": fake_mpi_mod,
        }):
            result = par._ensure_mpi()
            assert result is True
            assert par._HAS_MPI is True
            assert par.MPI is fake_mpi_mod

        par._HAS_MPI = old_has
        par.MPI = old_mpi

    def test_auto_detects_mpi(self) -> None:
        """
        Verify that 'auto' resolves to 'mpi' when an MPI environment variable is set and mpi4py is available. This test sets OMPI_COMM_WORLD_SIZE to simulate an MPI launcher environment and mocks _ensure_mpi to return True. The resulting backend string must be 'mpi' to confirm that auto-detection correctly promotes to MPI when the prerequisites are satisfied.

        Returns:
            None
        """
        import modvx.parallel as par

        fake_comm = MagicMock()
        fake_comm.Get_size.return_value = 4

        fake_mpi = MagicMock()
        fake_mpi.COMM_WORLD = fake_comm

        old_has = par._HAS_MPI
        old_mpi = par.MPI

        par._HAS_MPI = True
        par.MPI = fake_mpi

        with patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "4"}):
            with patch("modvx.parallel._ensure_mpi", return_value=True):
                result = par.ParallelProcessor._resolve_backend("auto")
                assert result == "mpi"

        par._HAS_MPI = old_has
        par.MPI = old_mpi


# -----------------------------------------------------------------------
# MPI backend constructor + _run_mpi
# -----------------------------------------------------------------------


class TestMpiBackend:
    """Cover MPI constructor init and _run_mpi execution."""

    def _make_mpi_processor(
        self,
        rank: int = 0,
        size: int = 2,
    ) -> tuple:
        """
        Build a ParallelProcessor configured with a mocked MPI communicator for MPI backend tests. The method temporarily replaces the module-level MPI and _HAS_MPI values with fakes, constructs a ParallelProcessor with the mpi backend, and then restores the originals. The returned tuple contains the processor instance and its mock execution function.

        Parameters:
            rank (int): Simulated MPI rank of the current process. Defaults to 0.
            size (int): Simulated MPI communicator size (total number of ranks). Defaults to 2.

        Returns:
            tuple: Two-element tuple (ParallelProcessor, MagicMock) containing the processor and its fn.
        """
        import modvx.parallel as par

        fn = MagicMock()

        fake_comm = MagicMock()
        fake_comm.Get_rank.return_value = rank
        fake_comm.Get_size.return_value = size

        fake_mpi = MagicMock()
        fake_mpi.COMM_WORLD = fake_comm

        old_has = par._HAS_MPI
        old_mpi = par.MPI
        par._HAS_MPI = True
        par.MPI = fake_mpi

        pp = par.ParallelProcessor(fn, backend="mpi")

        par._HAS_MPI = old_has
        par.MPI = old_mpi

        return pp, fn

    def test_mpi_constructor(self) -> None:
        """
        Verify that a ParallelProcessor built with the mpi backend correctly sets rank, size, comm, and _backend. This test uses the _make_mpi_processor helper to simulate a two-rank MPI environment and asserts all four attributes reflect the mocked communicator values. Correct attribute assignment is required for is_root, rank-based work distribution, and barrier synchronisation to work.

        Returns:
            None
        """
        pp, _ = self._make_mpi_processor(rank=0, size=4)
        assert pp.rank == 0
        assert pp.size == 4
        assert pp.comm is not None
        assert pp._backend == "mpi"

    def test_run_mpi_dispatches(self) -> None:
        """
        Verify that _run_mpi calls the execution function at least once and calls Barrier after processing. This test creates a two-rank processor as rank 0 and passes two work units. The execution function mock count must be at least 1 since rank 0 receives at least one group, and Barrier must be called exactly once to synchronise before the method returns.

        Returns:
            None
        """
        pp, fn = self._make_mpi_processor(rank=0, size=2)
        units = [
            {"cycle_start": datetime.datetime(2024, 9, 17), "region_name": "GLOBAL"},
            {"cycle_start": datetime.datetime(2024, 9, 18), "region_name": "GLOBAL"},
        ]
        pp._execute_mpi(units)
        assert fn.call_count >= 1
        pp.comm.Barrier.assert_called_once()

    def test_run_dispatches_mpi_backend(self) -> None:
        """
        Verify that calling the public run() method on an MPI-backend processor delegates to _run_mpi. This test confirms end-to-end dispatch from the public interface down to the MPI-aware execution path. The execution function mock count must be at least 1 for rank 0 handling one work unit.

        Returns:
            None
        """
        pp, fn = self._make_mpi_processor(rank=0, size=2)
        units = [
            {"cycle_start": datetime.datetime(2024, 9, 17), "region_name": "GLOBAL"},
        ]
        pp.run(units)
        assert fn.call_count >= 1

    def test_run_mpi_non_root_rank(self) -> None:
        """
        Verify that a non-root MPI rank still calls Barrier after processing its assigned work units. Barrier synchronisation must occur on all ranks, not just rank 0, to prevent the root from advancing past the barrier while workers are still running. This test uses rank 1 in a two-rank job and confirms Barrier is called once regardless of which units were assigned.

        Returns:
            None
        """
        pp, fn = self._make_mpi_processor(rank=1, size=2)
        units = [
            {"cycle_start": datetime.datetime(2024, 9, 17), "region_name": "GLOBAL"},
            {"cycle_start": datetime.datetime(2024, 9, 18), "region_name": "GLOBAL"},
        ]
        pp._execute_mpi(units)
        pp.comm.Barrier.assert_called_once()
