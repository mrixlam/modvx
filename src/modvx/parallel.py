#!/usr/bin/env python3

"""
Parallel execution for MODvx.

This module defines the ParallelProcessor class, which provides an interface for distributing and executing modvx work-units across multiple processes or MPI ranks. The class supports three execution backends: serial (single process), multiprocessing (using Python's multiprocessing library), and MPI (using mpi4py). The backend is automatically selected based on the environment and user configuration, with a preference for MPI when available. The ParallelProcessor handles the grouping of work-units by forecast cycle to maximize cache reuse, and it provides methods for executing the assigned units in each backend. The module also includes internal helper functions for managing multiprocessing worker initialization and execution, as well as lazy detection of MPI availability to avoid unnecessary imports in non-MPI contexts. 

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

# Initially set to None; will be assigned True/False on first call to _ensure_mpi.
_HAS_MPI: Optional[bool] = None 

# Initially set to None; will be assigned the mpi4py.MPI module if available. 
MPI = None  


def _ensure_mpi() -> bool:
    """
    This internal helper function checks for the availability of the mpi4py library, which is required for MPI-based parallel execution. It uses a module-level cache variable to store the availability status after the first check, so that subsequent calls can return immediately without attempting to import again. If mpi4py is not available, it sets the cache to False and returns False; if it is available, it imports the MPI module, assigns it to the global MPI variable, sets the cache to True, and returns True. This function should be called at the beginning of any code path that relies on MPI to ensure that the necessary dependencies are present before proceeding. 

    Parameters:
        None

    Returns:
        bool: True if mpi4py.MPI was successfully imported, False otherwise.
    """
    # Declare the module-level variables as global to allow assignment within this function.
    global _HAS_MPI, MPI

    # Check the cached availability flag to avoid repeated import attempts.
    if _HAS_MPI is not None:
        return _HAS_MPI
    
    # Identify whether mpi4py is available by attempting to import it. 
    try:
        from mpi4py import MPI as _MPI  # type: ignore[import-untyped]
        MPI = _MPI
        _HAS_MPI = True
    except ImportError:
        _HAS_MPI = False

    # Return the cached availability flag 
    return _HAS_MPI


# Module-level variable to store the execution function for multiprocessing workers.
_MP_EXECUTE_FN: Optional[Callable[[Dict[str, Any]], None]] = None


def _initialize_multiprocessing_worker(execute_fn: Callable[[Dict[str, Any]], None]) -> None:
    """
    This internal helper function is used as the initializer for multiprocessing worker processes. It takes the work-unit execution function as an argument and stores it in a module-level variable (_MP_EXECUTE_FN) that can be accessed by the worker processes when they execute their assigned work-units. This approach avoids the need to pickle the execution function with each task, which can be inefficient, and instead allows it to be set once per worker process at initialization. The worker execution function will be called with individual work-unit dictionaries when the workers are processing their assigned tasks. 

    Parameters:
        execute_fn (Callable[[Dict[str, Any]], None]): The work-unit execution function to store globally in the worker process.

    Returns:
        None
    """
    # Declare the module-level variable as global to allow assignment within this function.
    global _MP_EXECUTE_FN

    # Store the provided execution function in the module-level variable for use by worker processes.
    _MP_EXECUTE_FN = execute_fn


def _multiprocessing_worker_execute_units(units: List[Dict[str, Any]]) -> int:
    """
    This internal helper function is called by each multiprocessing worker process to execute its assigned list of work-unit dictionaries sequentially. It first asserts that the execution function has been initialized in the worker process (which should have been done by the initializer), then iterates through the list of assigned units and calls the execution function on each one. After processing all units, it returns the count of completed units, which can be used for logging or verification purposes. This function serves as the main execution loop for each worker process when using the multiprocessing backend. 

    Parameters:
        units (list of dict): Ordered list of work-unit dictionaries to execute sequentially.

    Returns:
        int: Number of work-units completed (equal to len(units) on success).
    """
    # Ensure that the execution function has been initialized in this worker process before attempting to call it.
    assert _MP_EXECUTE_FN is not None, "Worker not initialized"

    # Execute each unit in the assigned list sequentially
    for unit in units:
        _MP_EXECUTE_FN(unit)

    # Return the count of completed units for logging purposes
    return len(units)


class ParallelProcessor:
    """ Distribute and execute modvx work-units across multiple processes or MPI ranks. """

    def __init__(self: "ParallelProcessor",
                 execute_fn: Callable[[Dict[str, Any]], None],
                 backend: str = "auto",
                 nprocs: Optional[int] = None,) -> None:
        """
        This constructor initializes the ParallelProcessor with the provided execution function and backend configuration. It stores the execution function for later use in the worker processes, resolves the backend selection based on the requested value and environment conditions (with automatic detection of MPI), and sets up any necessary state for MPI execution if that backend is selected. The number of processes for multiprocessing can be specified or will default to the number of CPU cores available. This setup allows the run method to later dispatch work-units to the appropriate execution method based on the resolved backend. 
        
        Parameters:
            execute_fn (callable): Function that takes a single work-unit dict and performs the necessary processing. 
            backend (str): Execution backend to use: ``"auto"``, ``"mpi"``, ``"multiprocessing"``, or ``"serial"``. 
            nprocs (int or None): Number of processes to use for multiprocessing; ignored for other backends. 

        Returns:
            None
        """
        # Store the execution function for use in the various backend execution methods.
        self.execute_fn = execute_fn

        # Store the requested number of processes for multiprocessing
        self._nprocs = nprocs or os.cpu_count() or 1

        # Resolve the backend selection based on the requested value and environment conditions. 
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


    @staticmethod
    def _resolve_backend(requested: str) -> str:
        """
        This static method determines the concrete execution backend to use based on the user's request and the runtime environment. If the user explicitly requests a specific backend (``"mpi"``, ``"multiprocessing"``, or ``"serial"``), that backend is returned directly. If the user requests ``"auto"``, the method checks for environment variables commonly set by MPI launchers to infer if the code is running in an MPI context. If MPI hints are detected, it attempts to import mpi4py and checks if we're in a multi-rank context; if so, it selects the MPI backend. If no MPI environment is detected or if mpi4py cannot be imported, it falls back to serial execution. This logic allows for seamless operation across different environments without requiring users to manually specify the backend in most cases. 

        Parameters:
            requested (str): Backend name as requested by the user; ``"auto"`` triggers detection logic.

        Returns:
            str: Concrete backend name: ``"mpi"``, ``"multiprocessing"``, or ``"serial"``.
        """
        # If the user explicitly requested a specific backend, return it directly without detection.
        if requested != "auto":
            return requested

        # List of environment variables commonly set by MPI launchers that indicate an MPI execution context. 
        mpi_env_hints = (
            "OMPI_COMM_WORLD_SIZE",  # Open MPI
            "PMI_SIZE",             # MPICH
            "MPI_LOCALNRANKS",      # Intel MPI
            "PMIX_RANK",            # PMIx
        )

        # Determine if any of the known MPI environment variables are set, which suggests an MPI launch.
        launched_via_mpi = any(os.environ.get(k) for k in mpi_env_hints)

        # If MPI launch hints are present, attempt to import mpi4py and check if we're in a multi-rank context. 
        if launched_via_mpi and _ensure_mpi():
            assert MPI is not None
            if MPI.COMM_WORLD.Get_size() > 1:
                return "mpi"

        # Use serial execution if no MPI environment is detected or if mpi4py import fails
        return "serial"


    @staticmethod
    def _unit_cycle_key(unit: Dict[str, Any]) -> str:
        """
        This static helper function generates a string key for grouping work-units based on their forecast cycle, which is determined by the ``cycle_start`` field in the unit dictionary. The cycle key is formatted as a string in the form ``"<YYYYmmddHH>"``, which allows for easy grouping of units that belong to the same forecast cycle. This grouping is important for maximizing cache reuse when processing units from the same cycle, as they are likely to access similar data. The function assumes that the ``cycle_start`` field is a datetime object and formats it accordingly to produce the cycle key. 

        Parameters:
            unit (dict): Work-unit dictionary containing at least ``cycle_start``.

        Returns:
            str: Cycle key string in the format ``"<YYYYmmddHH>"``.
        """
        # Format the cycle_start datetime as a string key for grouping. 
        return unit["cycle_start"].strftime("%Y%m%d%H")


    @staticmethod
    def _group_units_by_cycle(units: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        This static helper function takes a list of work-unit dictionaries and groups them into a dictionary where the keys are cycle keys (derived from the ``cycle_start`` field of each unit) and the values are lists of units that belong to that cycle. This grouping allows for efficient scheduling of work-units in parallel backends while maximizing cache reuse, as units from the same cycle are likely to access similar data. The function iterates through all provided units, generates the appropriate cycle key for each unit, and appends it to the corresponding list in a defaultdict. Finally, it returns a regular dictionary with string keys for use in scheduling. 

        Parameters:
            units (list of dict): Flat list of all work-unit dictionaries.

        Returns:
            Dict[str, list of dict]: Dictionary mapping cycle keys to lists of work-units in that group.
        """
        # Initialize a defaultdict to accumulate units for each cycle key. 
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Iterate over all units and append them to the appropriate cycle group based on their cycle_start time.
        for unit in units:
            key = unit["cycle_start"].strftime("%Y%m%d%H")
            groups[key].append(unit)

        # Return a regular dict with string keys for cycle groups
        return groups


    def _assign_groups_to_workers_round_robin(self: "ParallelProcessor", 
                                              units: List[Dict[str, Any]], 
                                              num_workers: int,) -> Dict[int, List[Dict[str, Any]]]:
        """
        This internal helper function takes the complete list of work-unit dictionaries and the number of target workers (MPI ranks or multiprocessing pool size) and assigns the units to workers in a round-robin fashion based on their cycle groups. It first groups the units by their cycle keys using the _group_units_by_cycle method, then iterates through the sorted groups and assigns each group to a worker index in a round-robin manner. This ensures that all units from the same cycle are processed by the same worker, which can improve cache performance when accessing shared data. The resulting assignment is a dictionary mapping from zero-based worker indices to lists of assigned work-unit dictionaries. 

        Parameters:
            units (list of dict): Complete list of work-unit dictionaries to distribute.
            num_workers (int): Number of target workers (MPI ranks or process pool size).

        Returns:
            Dict[int, list of dict]: Mapping from zero-based worker index to assigned work-unit list.
        """
        # Extract groups of units that share the same cycle key, which allows for cache reuse within each group.
        groups = self._group_units_by_cycle(units)

        # Initialize a defaultdict to accumulate assigned units for each worker index. 
        assignment: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        # Assign groups to workers in a round-robin fashion based on sorted group keys 
        for group_index, (_, group_units) in enumerate(sorted(groups.items())):
            worker = group_index % num_workers
            assignment[worker].extend(group_units)

        # Return a regular dict with integer keys for worker indices 
        return assignment


    def run(self: "ParallelProcessor", 
            units: List[Dict[str, Any]]) -> None:
        """
        This method executes the provided list of work-unit dictionaries using the configured parallel backend. It first logs the start of execution and records the wall-clock time, then dispatches the units to the appropriate execution method based on the resolved backend (MPI, multiprocessing, or serial). After all units have been processed, it measures the total elapsed time and logs it on the root process. In MPI mode, a barrier is used to ensure that all ranks have completed before logging the final message. This method serves as the main entry point for executing work-units in parallel and abstracts away the details of task distribution and synchronization across different backends. 

        Parameters:
            units (list of dict): Complete list of work-unit dictionaries to execute.

        Returns:
            None
        """
        # Log the start of execution 
        t0 = time.time()

        # Select and execute the appropriate backend runner based on the resolved backend name.
        if self._backend == "mpi":
            self._execute_mpi(units)
        elif self._backend == "multiprocessing":
            self._execute_multiprocessing(units)
        else:
            self._execute_serial(units)

        # Measure total elapsed time and log it on the root process. 
        elapsed = time.time() - t0

        # Only the root process logs the total execution time to avoid redundant messages in MPI mode.
        if self.is_root:
            logger.info(
                "Total wall-clock time: %.1f s (%.1f min) [backend=%s]",
                elapsed, elapsed / 60.0, self._backend,
            )


    def _execute_serial(self: "ParallelProcessor", 
                        units: List[Dict[str, Any]]) -> None:
        """
        This method executes the provided list of work-unit dictionaries sequentially in the current process. It iterates through the list of units and calls the execution function on each one in order. This is the simplest execution mode and is used when no parallel backend is selected or when running in an environment without MPI support. The method logs the total number of work-units being executed for visibility, but does not perform any parallel distribution or synchronization since all work is done in a single process. 

        Parameters:
            units (list of dict): Complete list of work-unit dictionaries to execute in order.

        Returns:
            None
        """
        # Log the total number of work-units being executed in serial mode for visibility.
        logger.info("Running %d work-units in serial mode.", len(units))

        # Execute each unit sequentially in the current process.
        for unit in units:
            self.execute_fn(unit)


    def _execute_mpi(self: "ParallelProcessor", 
                     units: List[Dict[str, Any]]) -> None:
        """
        This method executes the provided list of work-unit dictionaries across multiple MPI ranks. It first groups the units by their forecast cycle and assigns them to ranks in a round-robin fashion to maximize cache reuse. Each rank retrieves its assigned subset of work-units from the assignment dictionary and executes them sequentially using the provided execution function. After all ranks have completed their assigned work, a barrier is used to synchronize before logging the completion message on the root rank. This method relies on mpi4py for MPI communication and assumes that the ParallelProcessor was initialized with the MPI backend and that the necessary environment variables for MPI were detected. 

        Parameters:
            units (list of dict): Complete list of work-unit dictionaries; each rank will receive and execute its assigned subset.

        Returns:
            None
        """
        # Group units by cycle and assign to ranks in a round-robin fashion to balance load while preserving cache locality.
        assignment = self._assign_groups_to_workers_round_robin(units, self.size)

        # Each rank retrieves its assigned work-units from the assignment dictionary
        my_units = assignment.get(self.rank, [])

        # Log the number of work-units assigned to this rank for visibility into load distribution.
        logger.info(
            "Rank %d/%d: executing %d work-units.",
            self.rank,
            self.size,
            len(my_units),
        )

        # Each rank executes its assigned units sequentially
        for unit in my_units:
            self.execute_fn(unit)

        # Synchronise all ranks before logging completion
        if self.comm is not None:
            self.comm.Barrier()

        # Only the root rank logs the completion message after all ranks have reached the barrier.
        if self.rank == 0:
            logger.info("All %d MPI ranks completed.", self.size)


    def _execute_multiprocessing(self: "ParallelProcessor", 
                                 units: List[Dict[str, Any]]) -> None:
        """
        This method executes the provided list of work-unit dictionaries across multiple processes using Python's multiprocessing library. It first determines the number of worker processes to use based on the configured number of processes and the total number of work-units. Then, it groups the units by their forecast cycle and assigns them to worker indices in a round-robin fashion to maximize cache reuse. A multiprocessing pool is created with the specified number of processes, and each worker process is initialized with the execution function. The pool's map method is used to dispatch the assigned lists of work-units to each worker, which executes them sequentially using the _multiprocessing_worker_execute_units helper function. After all workers have completed their tasks, the method logs the total number of completed units for visibility. This method abstracts away the details of process management and task distribution when using multiprocessing as the backend. 

        Parameters:
            units (list of dict): Complete list of work-unit dictionaries to distribute across the worker pool.

        Returns:
            None
        """
        # Determine the number of worker processes to use
        nprocs = min(self._nprocs, len(units))

        # Group units by cycle and assign to workers in a round-robin fashion to maximize cache reuse within each worker.
        assignment = self._assign_groups_to_workers_round_robin(units, nprocs)

        # Log the total number of work-units, worker count, and group count 
        logger.info(
            "Running %d work-units across %d worker processes (%d groups).",
            len(units),
            nprocs,
            len(self._group_units_by_cycle(units)),
        )

        # Create a list of work-unit lists for each worker index, ensuring that workers with no assigned units receive an empty list.
        work_packets = [assignment.get(i, []) for i in range(nprocs)]

        # Each worker will call _multiprocessing_worker_execute_units with its assigned list of units and execute them sequentially
        with mp.Pool(
            processes=nprocs,
            initializer=_initialize_multiprocessing_worker,
            initargs=(self.execute_fn,),
        ) as pool:
            counts = pool.map(_multiprocessing_worker_execute_units, work_packets)

        # Log total completed units and worker count after all processes finish
        logger.info(
            "Multiprocessing complete — %d work-units finished by %d workers.",
            sum(counts),
            nprocs,
        )


    @property
    def is_root(self: "ParallelProcessor") -> bool:
        """
        This property returns True if the current process is the root process, which is defined as MPI rank 0 in MPI mode or simply the single process in non-MPI modes. This is useful for determining whether to perform certain actions (like logging) that should only be done by one process to avoid redundant output. In MPI mode, only the process with rank 0 will return True; in serial or multiprocessing modes, all processes will return True since there is effectively only one "rank". 

        Parameters:
            None

        Returns:
            bool: True if this process is MPI rank 0 or running in a non-MPI mode.
        """
        # In MPI mode, only rank 0 is the root
        return self.rank == 0
