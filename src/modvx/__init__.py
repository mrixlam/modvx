#!/usr/bin/env python3

"""
Package entry points and convenience imports for MODvx.

This module exposes the primary public classes and helpers that make up the MODvx verification toolkit. It provides a concise, importable surface for constructing configurations, building work-units, running parallel processing backends, performing I/O and data validation, computingperformance metrics, and producing plots. Importing from this module is the recommended way for downstream tools to access the canonical API.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

__version__ = "1.0.0"

from modvx.config import ModvxConfig, load_config_from_yaml
from modvx.task_manager import TaskManager
from modvx.parallel import ParallelProcessor
from modvx.file_manager import FileManager
from modvx.data_validator import DataValidator
from modvx.perf_metrics import PerfMetrics
from modvx.visualizer import Visualizer

__all__ = [
    "ModvxConfig",
    "load_config_from_yaml",
    "TaskManager",
    "ParallelProcessor",
    "FileManager",
    "DataValidator",
    "PerfMetrics",
    "Visualizer",
]
