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
from modvx.parallel import (
    ParallelProcessor,
    _initialize_multiprocessing_worker,
    _multiprocessing_worker_execute_units,
)


def _make_unit(cycle: str, 
               region: str) -> dict:
    """
    This helper function creates a work-unit dictionary with the required keys for testing. It converts the cycle string into a datetime object and assigns the region name directly. This standardised format allows tests to focus on the grouping and execution logic without worrying about the specifics of unit construction. 

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


class TestResolveBackend:
    """ Tests for _resolve_backend static method. """

    def test_explicit_serial(self: "TestResolveBackend") -> None:
        """
        This test verifies that when the backend is explicitly set to 'serial', the _resolve_backend method returns 'serial' without modification. Explicitly requesting the serial backend should bypass any auto-detection logic and return the string as-is, ensuring that users can force serial execution regardless of environment. This guards against accidental promotion to multiprocessing or MPI when the user has specified serial.

        Parameters:
            None

        Returns:
            None
        """
        assert ParallelProcessor._resolve_backend("serial") == "serial"

    def test_explicit_multiprocessing(self: "TestResolveBackend") -> None:
        """
        This test verifies that when the backend is explicitly set to 'multiprocessing', the _resolve_backend method returns 'multiprocessing' without modification. Explicitly requesting multiprocessing should bypass any auto-detection logic and return the string as-is, allowing users to force multiprocessing execution even if MPI is available. This ensures that users have control over the parallelism model regardless of the environment. 

        Parameters:
            None

        Returns:
            None
        """
        assert ParallelProcessor._resolve_backend("multiprocessing") == "multiprocessing"

    def test_explicit_mpi(self: "TestResolveBackend") -> None:
        """
        This test verifies that when the backend is explicitly set to 'mpi', the _resolve_backend method returns 'mpi' without modification. Explicitly requesting MPI should bypass any auto-detection logic and return the string as-is, allowing users to force MPI execution even if the auto-detection would have chosen a different backend. This ensures that users have control over the parallelism model regardless of the environment.

        Parameters:
            None

        Returns:
            None
        """
        assert ParallelProcessor._resolve_backend("mpi") == "mpi"

    def test_auto_defaults_serial(self: "TestResolveBackend") -> None:
        """
        This test verifies that when the backend is set to 'auto' and no MPI environment variables are present, the _resolve_backend method defaults to 'serial'. In an environment without MPI indicators, auto-detection should fall back to serial execution to ensure compatibility. This test clears the environment variables and confirms that 'auto' resolves to 'serial', confirming the fallback logic works as intended.

        Parameters:
            None

        Returns:
            None
        """
        saved_env = dict(os.environ)
        os.environ.clear()
        try:
            result = ParallelProcessor._resolve_backend("auto")
            assert result == "serial"
        finally:
            os.environ.clear()
            os.environ.update(saved_env)


class TestGrouping:
    """ Tests for _group_key, _build_groups, and _assign_groups_round_robin. """

    def test_group_key(self: "TestGrouping") -> None:
        """
        This test verifies that the _group_key method generates a consistent key for grouping work units by cycle. The key should be derived solely from the cycle_start datetime of the unit, formatted as 'YYYYMMDDhh'. This ensures that all units from the same cycle are grouped together regardless of region, which is critical for maximizing cache reuse. The test creates a unit with a known cycle and checks that the generated key matches the expected string format. 

        Parameters:
            None

        Returns:
            None
        """
        unit = _make_unit("2024091700", "GLOBAL")
        key = ParallelProcessor._unit_cycle_key(unit)
        assert key == "2024091700"

    def test_build_groups(self: "TestGrouping") -> None:
        """
        This test verifies that the _group_units_by_cycle method correctly groups work units into a dictionary keyed by cycle. Units with the same cycle_start should be grouped together regardless of region, while units with different cycles should be in separate groups. The test creates three units, two sharing the same cycle and one with a different cycle, and checks that the resulting groups dictionary has the correct number of keys and that the units are grouped as expected. 

        Parameters:
            None

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

    def test_round_robin_assignment(self: "TestGrouping") -> None:
        """
        This test verifies that the _assign_groups_to_workers_round_robin method distributes groups of work units across the specified number of workers in a round-robin fashion. Given a list of work units grouped by cycle, the method should assign them to worker buckets such that the distribution is as even as possible. The test creates three units with different cycles and assigns them to two workers, then checks that the total number of assigned units matches the input and that the distribution follows a round-robin pattern (e.g., worker 0 gets 2 units, worker 1 gets 1 unit). 

        Parameters:
            None

        Returns:
            None
        """
        units = [
            _make_unit("2024091700", "GLOBAL"),
            _make_unit("2024091800", "GLOBAL"),
            _make_unit("2024091900", "GLOBAL"),
        ]
        fn = lambda *a, **kw: None  # noqa: E731
        pp = ParallelProcessor(fn, backend="serial")
        assignment = pp._assign_groups_to_workers_round_robin(units, 2)
        # 3 groups → worker 0 gets 2, worker 1 gets 1 (round robin)
        total = sum(len(v) for v in assignment.values())
        assert total == 3

    def test_same_group_same_worker(self: "TestGrouping") -> None:
        """
        This test verifies that work units from the same cycle are always assigned to the same worker bucket by the _assign_groups_to_workers_round_robin method. Units sharing the same cycle should not be split across different workers, as this would lead to redundant data loading and reduced cache efficiency. The test creates three units, two of which share the same cycle, and assigns them to two workers. It then checks that the two units with the same cycle are in the same worker's assigned list, confirming that grouping by cycle is respected in the assignment. 

        Parameters:
            None

        Returns:
            None
        """
        units = [
            _make_unit("2024091700", "GLOBAL"),
            _make_unit("2024091700", "TROPICS"),
            _make_unit("2024091800", "NAMERICA"),
        ]
        fn = lambda *a, **kw: None  # noqa: E731
        pp = ParallelProcessor(fn, backend="serial")
        assignment = pp._assign_groups_to_workers_round_robin(units, 2)
        # Both cycle-1700 units (GLOBAL + TROPICS) should be in the same worker
        for worker_units in assignment.values():
            cycles = {u["cycle_start"].strftime("%Y%m%d%H") for u in worker_units}
            assert len(cycles) == 1, (
                "Units from different cycles should not share a worker bucket"
            )


class TestSerialBackend:
    """ Tests for serial execution path. """

    def test_serial_executes_all_units(self: "TestSerialBackend") -> None:
        """
        This test verifies that the serial backend executes the provided function for every work unit in the input list. In serial mode, the ParallelProcessor should simply iterate over all units and call the execution function once per unit without any parallelism. The test creates a mock function and a list of two work units, runs them through a serial ParallelProcessor, and asserts that the mock was called exactly twice, confirming that all units were processed. 

        Parameters:
            None

        Returns:
            None
        """
        fn_calls: list = []

        def fn(unit):
            fn_calls.append(unit)

        pp = ParallelProcessor(fn, backend="serial")
        units = [
            _make_unit("2024091700", "GLOBAL"),
            _make_unit("2024091800", "GLOBAL"),
        ]
        pp.run(units)
        assert len(fn_calls) == 2

    def test_serial_order_preserved(self: "TestSerialBackend") -> None:
        """
        This test verifies that the serial backend processes work units in the order they are provided. In serial execution, the order of processing should match the input list, which can be important for debugging and for any side effects that depend on sequence. The test creates a mock function that tracks the region names of processed units, runs a list of two units with different regions through a serial ParallelProcessor, and asserts that the recorded calls match the input order, confirming that processing is sequential and ordered. 

        Parameters:
            None

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


class TestMpWorkerFunctions:
    """ Tests for _mp_init_worker and _mp_worker module-level functions. """

    def test_init_worker_sets_global(self: "TestMpWorkerFunctions") -> None:
        """
        This test verifies that the _mp_init_worker function correctly sets the global _MP_EXECUTE_FN variable to the provided function. This is critical for the multiprocessing worker processes to know which function to call when executing their assigned work units. The test creates a mock function, calls _mp_init_worker with it, and then asserts that the module-level _MP_EXECUTE_FN variable is set to the mock, confirming that the initialization logic correctly stores the execution function for later use by worker processes. 

        Parameters:
            None

        Returns:
            None
        """
        import modvx.parallel as par

        def fn(*a, **kw):
            pass

        _initialize_multiprocessing_worker(fn)
        assert par._MP_EXECUTE_FN is fn

    def test_mp_worker_executes(self: "TestMpWorkerFunctions") -> None:
        """
        This test verifies that the _mp_worker function calls the execution function set by _mp_init_worker for each work unit it receives. The worker function is the entry point for multiprocessing workers and must invoke the global execution function for each unit in its assigned batch. The test initializes the worker with a mock function, creates a list of one work unit, calls _mp_worker with that list, and asserts that the mock function was called once, confirming that the worker correctly executes its assigned units using the initialized function. 

        Parameters:
            None

        Returns:
            None
        """
        fn_calls: list = []

        def fn(unit):
            fn_calls.append(unit)

        _initialize_multiprocessing_worker(fn)
        units = [_make_unit("2024091700", "GLOBAL")]
        count = _multiprocessing_worker_execute_units(units)
        assert count == 1
        assert len(fn_calls) == 1


class TestMultiprocessingBackend:
    """ Tests for multiprocessing backend using mocked Pool. """

    def test_pool_called(self: "TestMultiprocessingBackend") -> None:
        """
        This test verifies that when the ParallelProcessor is configured with the multiprocessing backend, it creates a multiprocessing Pool and calls the map method to execute work units. The test uses unittest.mock to patch the Pool class and its context manager behavior, then runs a list of work units through the ParallelProcessor. Finally, it asserts that the map method was called once, confirming that the multiprocessing execution path is correctly invoked when the backend is set to multiprocessing. 

        Parameters:
            None

        Returns:
            None
        """
        import modvx.parallel as par

        map_calls: list = []

        class _FakeCtx:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def map(self, func, batches):
                map_calls.append(True)
                return [1, 1]

        fn_calls: list = []

        def fn(unit):
            fn_calls.append(unit)

        pp = ParallelProcessor(fn, backend="multiprocessing", nprocs=2)

        units = [
            _make_unit("2024091700", "GLOBAL"),
            _make_unit("2024091800", "GLOBAL"),
        ]

        orig_mp = par.mp
        par.mp = types.SimpleNamespace(Pool=lambda *a, **kw: _FakeCtx())

        try:
            pp.run(units)
        finally:
            par.mp = orig_mp

        assert len(map_calls) == 1


class TestProperties:
    """ Tests for ParallelProcessor properties. """

    def test_is_root_serial(self: "TestProperties") -> None:
        """
        This test verifies that the is_root property of a ParallelProcessor configured with the serial backend returns True. In a serial execution context, there is only one process, which should be considered the root. The test creates a ParallelProcessor with the serial backend and asserts that is_root is True, confirming that the property correctly identifies the single process as the root in a non-parallel environment.

        Parameters:
            None

        Returns:
            None
        """
        fn = lambda *a, **kw: None  # noqa: E731
        pp = ParallelProcessor(fn, backend="serial")
        assert pp.is_root is True

    def test_rank_serial(self: "TestProperties") -> None:
        """
        This test verifies that the rank property of a ParallelProcessor configured with the serial backend returns 0. In a serial execution context, there is only one process, which should have a rank of 0. The test creates a ParallelProcessor with the serial backend and asserts that rank is 0, confirming that the property correctly identifies the single process as rank 0 in a non-parallel environment. 

        Parameters:
            None

        Returns:
            None
        """
        fn = lambda *a, **kw: None  # noqa: E731
        pp = ParallelProcessor(fn, backend="serial")
        assert pp.rank == 0
        assert pp.size == 1


class TestEnsureMpiBranches:
    """ Cover _ensure_mpi success/failure paths and auto-detection of MPI. """

    def test_resolve_backend_auto_with_mpi_env(self: "TestEnsureMpiBranches") -> None:
        """
        This test verifies that when the backend is set to 'auto' and MPI environment variables are present, the _resolve_backend method promotes to 'mpi' if _ensure_mpi returns True. The presence of MPI environment variables should trigger an attempt to use MPI, and if mpi4py is available, the backend should resolve to 'mpi'. The test simulates an MPI environment by setting OMPI_COMM_WORLD_SIZE and mocking _ensure_mpi to return True, then asserts that 'auto' resolves to 'mpi', confirming that auto-detection correctly promotes to MPI when the prerequisites are satisfied. 

        Parameters:
            None

        Returns:
            None
        """
        import modvx.parallel as par

        old = par._HAS_MPI
        par._HAS_MPI = None

        key = "OMPI_COMM_WORLD_SIZE"
        orig_val = os.environ.get(key)
        os.environ[key] = "4"
        orig_ensure_mpi = par._ensure_mpi
        par._ensure_mpi = lambda: False
        try:
            result = par.ParallelProcessor._resolve_backend("auto")
            assert result == "serial"
        finally:
            if orig_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig_val
            par._ensure_mpi = orig_ensure_mpi
            par._HAS_MPI = old

    def test_ensure_mpi_import_failure(self: "TestEnsureMpiBranches") -> None:
        """
        This test verifies that the _ensure_mpi function returns False and sets _HAS_MPI to False when mpi4py cannot be imported. The test simulates an import failure by patching sys.modules to remove mpi4py and its MPI submodule, then calls _ensure_mpi and asserts that it returns False and that _HAS_MPI is set to False. This confirms that the function correctly handles the case where mpi4py is not available, preventing attempts to use MPI features when the library is missing. 

        Parameters:
            None

        Returns:
            None
        """
        import modvx.parallel as par

        import sys

        old_has = par._HAS_MPI
        old_mpi = par.MPI
        par._HAS_MPI = None
        par.MPI = None

        _MISSING = object()
        keys = ["mpi4py", "mpi4py.MPI"]
        saved = {k: sys.modules.get(k, _MISSING) for k in keys}
        for k in keys:
            sys.modules[k] = None  # type: ignore[assignment]
        try:
            result = par._ensure_mpi()
            assert result is False
            assert par._HAS_MPI is False
        finally:
            for k in keys:
                if saved[k] is _MISSING:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = saved[k]
            par._HAS_MPI = old_has
            par.MPI = old_mpi

    def test_ensure_mpi_success(self: "TestEnsureMpiBranches") -> None:
        """
        This test verifies that the _ensure_mpi function returns True and sets _HAS_MPI to True when mpi4py is successfully imported. The test simulates a successful import by creating fake mpi4py and mpi4py.MPI modules with the necessary attributes, then calls _ensure_mpi and asserts that it returns True, that _HAS_MPI is set to True, and that the MPI variable is set to the fake MPI module. This confirms that the function correctly detects the presence of mpi4py and prepares the module-level variables for MPI usage. 

        Parameters:
            None

        Returns:
            None
        """
        import modvx.parallel as par

        old_has = par._HAS_MPI
        old_mpi = par.MPI
        par._HAS_MPI = None

        import sys

        class _FakeComm:
            def Get_size(self) -> int:
                return 4
            def Get_rank(self) -> int:
                return 0

        fake_comm = _FakeComm()
        fake_mpi_mod = types.ModuleType("mpi4py.MPI")
        fake_mpi_mod.COMM_WORLD = fake_comm  # type: ignore[attr-defined]

        fake_mpi4py = types.ModuleType("mpi4py")
        setattr(fake_mpi4py, "MPI", fake_mpi_mod)

        _MISSING = object()
        keys = ["mpi4py", "mpi4py.MPI"]
        saved = {k: sys.modules.get(k, _MISSING) for k in keys}
        try:
            sys.modules["mpi4py"] = fake_mpi4py
            sys.modules["mpi4py.MPI"] = fake_mpi_mod
            result = par._ensure_mpi()
            assert result is True
            assert par._HAS_MPI is True
            assert par.MPI is fake_mpi_mod
        finally:
            for k in keys:
                if saved[k] is _MISSING:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = saved[k]
            par._HAS_MPI = old_has
            par.MPI = old_mpi

    def test_auto_detects_mpi(self: "TestEnsureMpiBranches") -> None:
        """
        This test verifies that when the backend is set to 'auto' and MPI environment variables are present, the _resolve_backend method promotes to 'mpi' if _ensure_mpi returns True. The presence of MPI environment variables should trigger an attempt to use MPI, and if mpi4py is available, the backend should resolve to 'mpi'. The test simulates an MPI environment by setting OMPI_COMM_WORLD_SIZE and mocking _ensure_mpi to return True, then asserts that 'auto' resolves to 'mpi', confirming that auto-detection correctly promotes to MPI when the prerequisites are satisfied. 

        Parameters:
            None

        Returns:
            None
        """
        import modvx.parallel as par

        class _FakeComm:
            def Get_size(self) -> int:
                return 4

        fake_comm = _FakeComm()
        fake_mpi = types.SimpleNamespace(COMM_WORLD=fake_comm)

        old_has = par._HAS_MPI
        old_mpi = par.MPI

        par._HAS_MPI = True
        par.MPI = fake_mpi

        key = "OMPI_COMM_WORLD_SIZE"
        orig_val = os.environ.get(key)
        os.environ[key] = "4"
        orig_ensure_mpi = par._ensure_mpi
        par._ensure_mpi = lambda: True
        try:
            result = par.ParallelProcessor._resolve_backend("auto")
            assert result == "mpi"
        finally:
            if orig_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig_val
            par._ensure_mpi = orig_ensure_mpi
            par._HAS_MPI = old_has
            par.MPI = old_mpi


class TestMpiBackend:
    """ Cover MPI constructor init and _run_mpi execution. """

    def _make_mpi_processor(self: "TestMpiBackend",
                            rank: int = 0,
                            size: int = 2,) -> tuple:
        """
        This helper method creates a ParallelProcessor instance configured with the MPI backend using mocked MPI communicator values. It simulates an MPI environment by creating fake MPI and communicator objects that return specified rank and size values. This allows tests to verify MPI-specific logic without requiring an actual MPI runtime. The method also mocks the execution function to track calls during tests. 

        Parameters:
            rank (int): Simulated MPI rank of the current process. Defaults to 0.
            size (int): Simulated MPI communicator size (total number of ranks). Defaults to 2.

        Returns:
            tuple: Two-element tuple (ParallelProcessor, MagicMock) containing the processor and its fn.
        """
        import modvx.parallel as par

        class _CallTracker:
            def __init__(self) -> None:
                self.call_count = 0
            def __call__(self, *a: object, **kw: object) -> None:
                self.call_count += 1

        fn = _CallTracker()

        class _FakeComm:
            def __init__(self, _rank: int, _size: int) -> None:
                self._rank = _rank
                self._size = _size
                self.barrier_calls = 0
            def Get_rank(self) -> int:
                return self._rank
            def Get_size(self) -> int:
                return self._size
            def Barrier(self) -> None:
                self.barrier_calls += 1

        fake_comm = _FakeComm(rank, size)
        fake_mpi = types.SimpleNamespace(COMM_WORLD=fake_comm)

        old_has = par._HAS_MPI
        old_mpi = par.MPI
        par._HAS_MPI = True
        par.MPI = fake_mpi

        pp = par.ParallelProcessor(fn, backend="mpi")

        par._HAS_MPI = old_has
        par.MPI = old_mpi

        return pp, fn

    def test_mpi_constructor(self: "TestMpiBackend") -> None:
        """
        This test verifies that the ParallelProcessor constructor correctly initializes the MPI backend with the provided rank and size values from the mocked MPI communicator. The test creates a processor with a specified rank and size, then asserts that the processor's rank, size, and comm attributes match the expected values, confirming that the constructor correctly sets up the MPI environment for use in parallel execution 

        Parameters:
            None

        Returns:
            None
        """
        pp, _ = self._make_mpi_processor(rank=0, size=4)
        assert pp.rank == 0
        assert pp.size == 4
        assert pp.comm is not None
        assert pp._backend == "mpi"

    def test_run_mpi_dispatches(self: "TestMpiBackend") -> None:
        """
        This test verifies that the _execute_mpi method of the ParallelProcessor correctly calls the execution function for the assigned work units and that it calls Barrier on the communicator after processing. The test simulates an MPI environment with rank 0 and size 2, creates a list of work units, and calls _execute_mpi. It then asserts that the execution function was called at least once (since rank 0 should have some work) and that the Barrier method was called once, confirming that the MPI execution path is correctly processing units and synchronizing across ranks. 

        Parameters:
            None

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
        assert pp.comm.barrier_calls == 1

    def test_run_dispatches_mpi_backend(self: "TestMpiBackend") -> None:
        """
        This test verifies that the run method of the ParallelProcessor correctly dispatches to the _execute_mpi method when the backend is set to 'mpi'. The test simulates an MPI environment with rank 0 and size 2, creates a list of work units, and calls run. It then asserts that the execution function was called at least once, confirming that the run method correctly routes to the MPI execution path when configured for MPI. 

        Parameters:
            None

        Returns:
            None
        """
        pp, fn = self._make_mpi_processor(rank=0, size=2)
        units = [
            {"cycle_start": datetime.datetime(2024, 9, 17), "region_name": "GLOBAL"},
        ]
        pp.run(units)
        assert fn.call_count >= 1

    def test_run_mpi_non_root_rank(self: "TestMpiBackend") -> None:
        """
        This test verifies that the _execute_mpi method of the ParallelProcessor correctly processes work units for a non-root rank and calls Barrier on the communicator. The test simulates an MPI environment with rank 1 and size 2, creates a list of work units, and calls _execute_mpi. It then asserts that the execution function was called at least once (since rank 1 should have some work) and that the Barrier method was called once, confirming that the MPI execution path is correctly processing units and synchronizing across ranks even for non-root ranks. 

        Parameters:
            None

        Returns:
            None
        """
        pp, fn = self._make_mpi_processor(rank=1, size=2)
        units = [
            {"cycle_start": datetime.datetime(2024, 9, 17), "region_name": "GLOBAL"},
            {"cycle_start": datetime.datetime(2024, 9, 18), "region_name": "GLOBAL"},
        ]
        pp._execute_mpi(units)
        assert pp.comm.barrier_calls == 1
