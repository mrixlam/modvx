"""
modvx — Model Verification Toolkit
====================================

A community Python package for computing spatial verification metrics
(Fractions Skill Score, RMSE, and more) for gridded forecast experiments
against satellite-based precipitation observations.

Classes
-------
TaskManager        Orchestrates work-unit enumeration, logging, and execution.
ParallelProcessor  MPI-based parallelism via mpi4py (graceful serial fallback).
FileManager        All I/O: forecast/observation loading, caching, output writing.
DataValidator      Grid preparation, regridding, mask application, QC checks.
PerfMetrics        Verification scores (FSS, RMSE, …).
Visualizer         Plotting: FSS-vs-lead-time, horizontal precip maps, diffs.
"""

__version__ = "0.1.0"

from modvx.config import ModvxConfig, load_config
from modvx.task_manager import TaskManager
from modvx.parallel import ParallelProcessor
from modvx.file_manager import FileManager
from modvx.data_validator import DataValidator
from modvx.perf_metrics import PerfMetrics
from modvx.visualizer import Visualizer

__all__ = [
    "ModvxConfig",
    "load_config",
    "TaskManager",
    "ParallelProcessor",
    "FileManager",
    "DataValidator",
    "PerfMetrics",
    "Visualizer",
]
