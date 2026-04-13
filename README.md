# MODvx — Model Verification Toolkit

[![CI](https://github.com/mrixlam/modvx/workflows/CI/badge.svg)](https://github.com/mrixlam/modvx/actions)
[![codecov](https://codecov.io/gh/mrixlam/modvx/branch/master/graph/badge.svg)](https://codecov.io/gh/mrixlam/modvx)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-alpha-orange.svg)

MODvx is a lightweight, modular Python toolkit for operational NWP model verification, built around the **Fraction Skill Score (FSS)** and a full suite of contingency-table metrics (POD, FAR, CSI, FBIAS, ETS). It provides a production-ready pipeline for computing, extracting, and visualising spatial verification metrics across multi-cycle forecast experiments, with native support for MPAS unstructured-mesh output, flexible parallelism, and reproducible YAML-driven configuration.

The pipeline accepts forecast NetCDF files (regular lat-lon or native MPAS mesh) and IMERG observation files, accumulates precipitation over configurable windows, regrids both fields to a common target grid, applies NetCDF region masks, and produces per-cycle NetCDF output files along with aggregated CSV tables and publication-quality plots — all driven by a single flat YAML configuration file.

## Key features

* **Full verification metric suite** — FSS (spatial), POD, FAR, CSI, FBIAS, and ETS (contingency-table), all NaN-aware and computed in a single optimized sweep
* **Batch-optimized computation** — binary exceedance masks are computed once per threshold and reused across all window sizes, eliminating redundant processing
* **Flexible precipitation accumulation** — configurable accumulation windows (`precip_accum_hours`) independent of the forecast output frequency
* **Native MPAS mesh support** — reads MPAS diagnostic files directly on the unstructured mesh via [MPASdiag](https://github.com/mrixlam/MPASdiag); no `convert-mpas` step required
* **Three parallel backends** — serial, Python `multiprocessing`, and MPI (via `mpi4py`), with automatic serial fallback when MPI is unavailable
* **NetCDF region masks** — define verification domains using arbitrary 2-D mask files; multiple domains can be verified in a single run
* **Configurable regridding** — regrid to the obs grid, the forecast grid, or a fixed resolution; supports both regular and MPAS-remapped grids
* **Per-experiment CSV extraction and batch plotting** — aggregated lead-time series plots for all metrics, domains, thresholds, and window sizes in one command
* **Reproducible YAML + CLI workflow** — all parameters live in a flat YAML file; any field can be overridden at runtime via CLI flags
* **Modular, testable design** — clear separation of concerns across seven focused classes, with a comprehensive pytest suite and coverage reporting

## Quick start

```bash
# Create and activate conda environment
conda create -n modvx python=3.10 -y
conda activate modvx

# Install dependencies
conda install -y -c conda-forge numpy scipy xarray matplotlib pyyaml pytest mpi4py netCDF4 ruff black pluggy coverage pytest-cov cartopy

# Install modvx in development mode
git clone https://github.com/mrixlam/modvx.git
cd modvx
pip install -e .

# Install mpasdiag (required — reads native MPAS mesh files)
git clone https://github.com/mrixlam/MPASdiag.git
pip install -e MPASdiag

# Run FSS computation with defaults
modvx run -c configs/default.yaml --vxdomain GLOBAL

# Run FSS computation with multiprocessing backend
modvx run -c configs/test.yaml --vxdomain GLOBAL --backend multiprocessing --nprocs 12

# Run with native MPAS mesh input (no convert-mpas needed)
modvx run -c configs/test.yaml --vxdomain GLOBAL

# Extract results to CSV
modvx extract-csv -c configs/test.yaml

# Generate plots for ALL metrics (fss, pod, far, csi, fbias, ets)
modvx plot --all -c configs/test.yaml

# Generate plots for specific metrics only
modvx plot --all -c configs/test.yaml --metric fss,pod,csi

# Single plot for a specific metric, domain, threshold, window
modvx plot -c configs/test.yaml --domain GLOBAL --thresh 90 --window 3 --metric ets

# List available options
modvx validate -c configs/test.yaml
```

## YAML configuration

MODvx uses flat YAML configuration files to specify all aspects of the verification workflow. All parameters live at the top level (no nested sections), making configs easy to read, diff, and override from the CLI. Copy `configs/default.yaml` as your starting point and edit the sections below.

```yaml
# configs/default.yaml  —  annotated reference configuration

# ---- experiment / time range -------------------------------------------

experiment_name: "mrislam_coldstart_240km_meso"  # used in output filenames and plot titles
initial_cycle_start: "20140901T00"               # first cycle (yyyymmddThh)
final_cycle_start:   "20140902T18"               # last cycle  (yyyymmddThh)
forecast_step_hours: 1         # interval between forecast output files (hours)
observation_interval_hours: 1  # accumulation window of the observation dataset (hours)
cycle_interval_hours: 6       # interval between initialisation cycles (hours)
forecast_length_hours: 12      # total lead time to verify (hours)
precip_accum_hours: 0          # 0 = same as forecast_step_hours; set to e.g. 3 for 3h accumulated precip

# ---- MPAS mesh settings (leave mpas_grid_file empty for regular-grid input) ----

mpas_grid_file: ""          # path to MPAS static grid file; empty = not an MPAS run
mpas_remap_resolution: 1.0  # target lat-lon resolution after MPAS remapping (degrees)

# ---- grid / resolution --------------------------------------------------

# "obs"  = regrid forecast to the observation grid (default)
# "fcst" = regrid observation to the forecast grid
# 0.25   = regrid both to a fixed resolution in degrees
target_resolution: "obs"

# ---- verification domains -----------------------------------------------

vxdomain:              # domains to verify (subset of keys defined in 'regions')
  - "GLOBAL"

regions:               # map of domain name → mask NetCDF filename (under mask_dir)
  SINGV:     "SINGV.nc"
  TROPICS:   "G004_TROPICS.nc"
  GLOBAL:    "G004_GLOBAL.nc"
  AFRICA:    "G004_AFRICA.nc"
  ASIA:      "G004_ASIA.nc"
  NAMERICA:  "G004_NAMERICA.nc"
  SAMERICA:  "G004_SAMERICA.nc"
  NHEM:      "G004_NHEM.nc"
  SHEM:      "G004_SHEM.nc"
  AUNZ:      "WAFS0P25_AUNZ.nc"

# ---- metric parameters --------------------------------------------------

thresholds:            # percentile thresholds for FSS and contingency metrics
  - 90.0
  - 95.0
  - 97.5
  - 99.0

window_sizes:          # spatial smoothing half-widths (grid points)
  - 1
  - 3
  - 5
  - 7
  - 9
  - 11
  - 13
  - 15

# "independent" = separate percentile thresholds computed for fcst and obs
# "obs_only"    = threshold derived from obs only, then applied to both fields
threshold_mode: "independent"

# ---- variable names -----------------------------------------------------

obs_var_name: "precip"   # variable name to read from observation NetCDF files

# ---- directories (all relative to base_dir unless absolute) -------------

base_dir:   "data"        # root directory; all other paths resolved against this
fcst_dir:   "fcst"        # forecast NetCDF files
obs_dir:    "obs"         # observation NetCDF files
mask_dir:   "masks"       # region mask NetCDF files
output_dir: "output"      # FSS result NetCDF files
debug_dir:  "debug"       # intermediate debug outputs
log_dir:    "logs"        # run logs (when enable_logs: true)
csv_dir:    "csv"         # extracted CSV tables
plot_dir:   "plots"       # generated plots

# ---- filename templates -------------------------------------------------

# Tag embedded in observation filenames after the vintage token
obs_file_tag: "V07B.SRCHHR.X360Y180.R1p0.FMT"

observation_template: >-
  {obs_dir}/IMERG.A01H.VLD{date_key}.S{date_key}T000000.E{date_key}T235959.{vintage}.{obs_file_tag}.nc

obs_vintage_preference:  # tried in order; first file found on disk is used
  - "FNL"
  - "LTE"

# ---- I/O tuning ---------------------------------------------------------

compression_level: 9    # zlib compression level for output NetCDF files (0–9)
clip_buffer_deg: 1.0    # degrees of padding when clipping data to a region mask

# ---- runtime flags ------------------------------------------------------

verbose: false            # print extra debug output to stdout
save_intermediate: false  # write intermediate regridded fields to debug_dir
enable_logs: false        # write per-cycle log files to log_dir
```

### Key concepts

| Parameter | Notes |
|---|---|
| `thresholds` | Percentile-based (e.g. `90.0` = 90th percentile), not absolute values. |
| `threshold_mode` | `"independent"` gives separate fcst/obs exceedance; `"obs_only"` uses a single obs-derived threshold for both. |
| `precip_accum_hours` | Set to `0` to use `forecast_step_hours` as the accumulation window. Must be a multiple of both `forecast_step_hours` and `observation_interval_hours`. |
| `target_resolution` | Use `"obs"` for most runs; `"fcst"` when the forecast grid is coarser than the obs. |
| `mpas_grid_file` | Set this to the MPAS static grid NetCDF when running with native MPAS diag files (no convert-mpas required). |
| `obs_file_tag` | Tag string embedded in IMERG observation filenames after the vintage token; override when using a different IMERG product version or resolution. |
| `vxdomain` | Must be a subset of the keys listed under `regions`. Pass `--vxdomain GLOBAL` on the CLI to override at runtime. |

## Package structure

```
modvx/
├── configs/default.yaml     # Default configuration
├── src/modvx/
│   ├── __init__.py          # Package entry point
│   ├── cli.py               # CLI (modvx run|extract-csv|plot|validate)
│   ├── config.py            # ModvxConfig dataclass + YAML loader
│   ├── utils.py             # Shared helpers
│   ├── task_manager.py      # Orchestration
│   ├── parallel.py          # MPI / multiprocessing / serial backends
│   ├── file_manager.py      # All file I/O and CSV extraction
│   ├── mpas_reader.py       # Native MPAS mesh reader (via MPASdiag)
│   ├── data_validator.py    # Grid prep pipeline (regrid, clip, mask)
│   ├── perf_metrics.py      # FSS + contingency-table metrics (POD, FAR, CSI, FBIAS, ETS)
│   └── visualizer.py        # Batch plotting
├── tests/                   # Unit tests
├── pyproject.toml           # Build configuration
└── environment.yml          # Conda environment
```

## Testing

Run unit tests with pytest:

```bash
pytest tests/
```

Run tests with coverage report:

```bash
pytest --cov=modvx tests/ --cov-report=term-missing
```

Run style checks with ruff:

```bash
ruff check src/modvx/ tests/
```

## License

MIT — see [pyproject.toml](pyproject.toml) for details.
