#!/usr/bin/env python3

"""
Parallel execution for modvx.

This module defines the ParallelProcessor class, which provides a unified interface for executing verification tasks in parallel using different backends such as multiprocessing or Dask. The ParallelProcessor abstracts away the details of task distribution, worker management, and result collection, allowing the rest of the modvx pipeline to focus on defining the work units and processing logic. By centralizing parallel execution logic, this module enables flexible scaling across different computing environments while maintaining a consistent API for task submission and monitoring. The ParallelProcessor is designed to work seamlessly with the TaskManager, which generates the work units to be processed in parallel.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from __future__ import annotations

import os
import time
import logging
import multiprocessing as mp
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MPI import — only performed when MPI backend is actually requested
# ---------------------------------------------------------------------------
_HAS_MPI: Optional[bool] = None  # None = not yet checked
MPI = None  # will be set to mpi4py.MPI if available


def _ensure_mpi() -> bool:
    """
    Lazily attempt to import mpi4py.MPI and cache the result for subsequent calls. The import is performed at most once regardless of how many times this function is called, using a module-level flag to avoid repeated import overhead. On success, the MPI module is stored in the module-level ``MPI`` variable for use by ParallelProcessor. This function does not raise on failure; callers are responsible for checking the return value.

    Returns:
        bool: True if mpi4py.MPI was successfully imported, False otherwise.
    """
    global _HAS_MPI, MPI
    if _HAS_MPI is not None:
        return _HAS_MPI
    try:
        from mpi4py import MPI as _MPI  # type: ignore[import-untyped]
        MPI = _MPI
        _HAS_MPI = True
    except ImportError:
        _HAS_MPI = False
    return _HAS_MPI


# ---------------------------------------------------------------------------
# Module-level worker state for multiprocessing backend
# ---------------------------------------------------------------------------
_MP_EXECUTE_FN: Optional[Callable[[Dict[str, Any]], None]] = None


def _initialize_multiprocessing_worker(execute_fn: Callable[[Dict[str, Any]], None]) -> None:
    """
    Initialise a multiprocessing worker process by storing the execution function globally.
    This function is passed as the ``initializer`` argument to ``multiprocessing.Pool`` and
    is called once in each worker process before any work units are dispatched. Storing
    the function in the module-level ``_MP_EXECUTE_FN`` variable allows worker processes to
    call it for each assigned unit without re-pickling the function for every task.

    Parameters:
        execute_fn (Callable[[Dict[str, Any]], None]): The work-unit execution function
            to store globally in the worker process.

    Returns:
        None
    """
    global _MP_EXECUTE_FN
    _MP_EXECUTE_FN = execute_fn


def _multiprocessing_worker_execute_units(units: List[Dict[str, Any]]) -> int:
    """
    Execute a list of work-units sequentially within a single multiprocessing worker
    process. The execution function must have been stored in the module-level
    ``_MP_EXECUTE_FN`` variable by a prior call to the pool initializer; an assertion
    error is raised if this precondition is not met. Grouping multiple units into a
    single worker call allows the FileManager's in-memory cache to be reused across
    units in the same (cycle, region) group, reducing redundant I/O.

    Parameters:
        units (list of dict): Ordered list of work-unit dictionaries to execute sequentially.

    Returns:
        int: Number of work-units completed (equal to len(units) on success).
    """
    assert _MP_EXECUTE_FN is not None, "Worker not initialised"
    for unit in units:
        _MP_EXECUTE_FN(unit)
    return len(units)


class ParallelProcessor:
    """
    Distribute and execute modvx work-units across multiple processes or MPI ranks. Three backends are supported: ``"mpi"`` for multi-node HPC environments via mpi4py, ``"multiprocessing"`` for single-machine parallelism via Python's multiprocessing pool, and ``"serial"`` for sequential debugging. Backend selection defaults to ``"auto"``, which inspects environment variables to detect MPI launch conditions and falls back to serial when no parallelism is detectable.

    Parameters:
        execute_fn (callable): Function accepting a single work-unit dict, typically
            TaskManager.execute_work_unit.
        backend (str): Execution backend: ``"auto"`` (default), ``"mpi"``,
            ``"multiprocessing"``, or ``"serial"``.
        nprocs (int or None): Worker count for the multiprocessing backend; defaults to
            os.cpu_count().
    """

    def __init__(
        self,
        execute_fn: Callable[[Dict[str, Any]], None],
        backend: str = "auto",
        nprocs: Optional[int] = None,
    ) -> None:
        self.execute_fn = execute_fn
        self._nprocs = nprocs or os.cpu_count() or 1
        self._backend = self._resolve_backend(backend)

        # MPI-specific state (only import mpi4py when actually needed)
        if self._backend == "mpi":
            _ensure_mpi()
            assert MPI is not None, "mpi4py required for MPI backend"
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

    # ------------------------------------------------------------------
    # Backend auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        """
        Resolve the ``"auto"`` backend selection to a concrete named backend. The selection logic checks well-known MPI launcher environment variables (for Open MPI, MPICH, Intel MPI, and PMIx) to determine whether the process was started under mpirun, avoiding an unconditional mpi4py import that would slow down non-MPI runs. If a multi-rank MPI context is detected, ``"mpi"`` is returned; otherwise ``"serial"`` is the default. Explicit backend names are returned unchanged.

        Parameters:
            requested (str): Backend name as requested by the user; ``"auto"`` triggers
                detection logic.

        Returns:
            str: Concrete backend name: ``"mpi"``, ``"multiprocessing"``, or ``"serial"``.
        """
        if requested != "auto":
            return requested

        # Check environment for MPI launch hints (avoids importing mpi4py
        # when not running under mpirun/mpiexec).
        mpi_env_hints = (
            "OMPI_COMM_WORLD_SIZE",  # Open MPI
            "PMI_SIZE",             # MPICH
            "MPI_LOCALNRANKS",      # Intel MPI
            "PMIX_RANK",            # PMIx
        )
        launched_via_mpi = any(os.environ.get(k) for k in mpi_env_hints)

        if launched_via_mpi and _ensure_mpi():
            assert MPI is not None
            if MPI.COMM_WORLD.Get_size() > 1:
                return "mpi"

        return "serial"

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unit_cycle_key(unit: Dict[str, Any]) -> str:
        """
        Generate a string key that identifies work-units sharing the same forecast cycle.
        Units with the same cycle start time access identical forecast files and can reuse
        cached data, so grouping by the cycle enables assigning them to the same worker.

        Parameters:
            unit (dict): Work-unit dictionary containing at least ``cycle_start``.

        Returns:
            str: Cycle key string in the format ``"<YYYYmmddHH>"``.
        """
        return unit["cycle_start"].strftime("%Y%m%d%H")

    @staticmethod
    def _group_units_by_cycle(units: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organise a flat list of work-units into groups keyed by forecast cycle. Grouping
        units by cycle enables each worker to reuse cached forecast data across regions
        within the same cycle, reducing redundant I/O.

        Parameters:
            units (list of dict): Flat list of all work-unit dictionaries.

        Returns:
            Dict[str, list of dict]: Dictionary mapping cycle keys to lists of work-units
                in that group.
        """
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for unit in units:
            key = unit["cycle_start"].strftime("%Y%m%d%H")
            groups[key].append(unit)
        return groups

    def _assign_groups_to_workers_round_robin(
        self, units: List[Dict[str, Any]], num_workers: int,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Assign work-unit groups to workers using a round-robin scheduling strategy. Groups are sorted by key before assignment, ensuring deterministic distribution across runs. All units within the same (cycle, region) group are assigned to the same worker index, preserving the data-locality property that enables the FileManager's in-memory cache to be reused across threshold/window combinations.

        Parameters:
            units (list of dict): Complete list of work-unit dictionaries to distribute.
            num_workers (int): Number of target workers (MPI ranks or process pool size).

        Returns:
            Dict[int, list of dict]: Mapping from zero-based worker index to assigned
                work-unit list.
        """
        groups = self._group_units_by_cycle(units)
        assignment: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for group_index, (_, group_units) in enumerate(sorted(groups.items())):
            worker = group_index % num_workers
            assignment[worker].extend(group_units)
        return assignment

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, units: List[Dict[str, Any]]) -> None:
        """
        Execute all work-units using the configured parallel backend. The appropriate backend runner (_run_serial, _run_mpi, or _run_multiprocessing) is selected based on the resolved backend name and dispatched with the full unit list. Total wall-clock time is measured and logged at INFO level on the root process after all units have completed. This is the primary public method called by the CLI run command after TaskManager.build_work_units returns the full task list.

        Parameters:
            units (list of dict): Complete list of work-unit dictionaries to execute.

        Returns:
            None
        """
        t0 = time.time()

        if self._backend == "mpi":
            self._execute_mpi(units)
        elif self._backend == "multiprocessing":
            self._execute_multiprocessing(units)
        else:
            self._execute_serial(units)

        elapsed = time.time() - t0
        if self.is_root:
            logger.info(
                "Total wall-clock time: %.1f s (%.1f min) [backend=%s]",
                elapsed, elapsed / 60.0, self._backend,
            )

    # ------------------------------------------------------------------
    # Serial backend
    # ------------------------------------------------------------------

    def _execute_serial(self, units: List[Dict[str, Any]]) -> None:
        """
        Execute all work-units sequentially in the current process. This is the simplest backend and requires no inter-process communication or synchronisation. It is used when no parallelism is available or desired, and is the safe fallback when neither MPI nor multiprocessing can be initialised. Useful for debugging because all execution occurs in a single Python scope.

        Parameters:
            units (list of dict): Complete list of work-unit dictionaries to execute in order.

        Returns:
            None
        """
        logger.info("Running %d work-units in serial mode.", len(units))
        for unit in units:
            self.execute_fn(unit)

    # ------------------------------------------------------------------
    # MPI backend
    # ------------------------------------------------------------------

    def _execute_mpi(self, units: List[Dict[str, Any]]) -> None:
        """
        Distribute and execute work-units across MPI ranks using round-robin group assignment. Each rank receives a disjoint subset of work-unit groups, with all units in the same (cycle, region) group guaranteed to land on the same rank to maximise cache reuse. A collective barrier at the end ensures all ranks have completed before the root process logs the completion message and control returns to the caller.

        Parameters:
            units (list of dict): Complete list of work-unit dictionaries; each rank will
                receive and execute its assigned subset.

        Returns:
            None
        """
        assignment = self._assign_groups_to_workers_round_robin(units, self.size)
        my_units = assignment.get(self.rank, [])

        logger.info(
            "Rank %d/%d: executing %d work-units.",
            self.rank,
            self.size,
            len(my_units),
        )

        for unit in my_units:
            self.execute_fn(unit)

        # Barrier so all ranks finish before any post-processing.
        if self.comm is not None:
            self.comm.Barrier()

        if self.rank == 0:
            logger.info("All %d MPI ranks completed.", self.size)


    # ------------------------------------------------------------------
    # Multiprocessing backend
    # ------------------------------------------------------------------

    def _execute_multiprocessing(self, units: List[Dict[str, Any]]) -> None:
        """
        Distribute and execute work-units across a pool of worker processes. Work-units are grouped by (cycle, region) and assigned round-robin to workers so that each worker's in-memory FileManager cache remains effective within its group. The pool is initialised with _mp_init_worker to install the execution function without repeated pickling, and pool.map blocks until all workers finish.

        Parameters:
            units (list of dict): Complete list of work-unit dictionaries to distribute
                across the worker pool.

        Returns:
            None
        """
        nprocs = min(self._nprocs, len(units))
        assignment = self._assign_groups_to_workers_round_robin(units, nprocs)

        logger.info(
            "Running %d work-units across %d worker processes (%d groups).",
            len(units),
            nprocs,
            len(self._group_units_by_cycle(units)),
        )

        # Build a list of per-worker unit-lists
        work_packets = [assignment.get(i, []) for i in range(nprocs)]

        with mp.Pool(
            processes=nprocs,
            initializer=_initialize_multiprocessing_worker,
            initargs=(self.execute_fn,),
        ) as pool:
            counts = pool.map(_multiprocessing_worker_execute_units, work_packets)

        logger.info(
            "Multiprocessing complete — %d work-units finished by %d workers.",
            sum(counts),
            nprocs,
        )

    # Note: legacy short-name aliases removed; use descriptive method names.

    @property
    def is_root(self) -> bool:
        """
        Indicate whether the current process is the root (rank 0) process. In MPI mode, only rank 0 returns True, which is used to gate log messages and post-processing steps that should execute only once across the communicator. In serial and multiprocessing modes, this property always returns True since there is only one controlling process.

        Returns:
            bool: True if this process is MPI rank 0 or running in a non-MPI mode.
        """
        return self.rank == 0
