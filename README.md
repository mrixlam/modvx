# MODvx — Model Verification Toolkit

MODvx is a lightweight, modular, and extensible Python package for model verification, with a focus on the **Fraction Skill Score (FSS)**. It provides a streamlined workflow for computing, extracting, and visualising FSS and other verification metrics across forecast experiments, designed to be flexible and user-friendly for the NWP modelling community. 

Key features include:
* YAML + CLI configuration for easy setup and reproducibility
* Optional MPI parallelism with graceful serial fallback
* Support for NetCDF region masks to define flexible verification domains
* Native MPAS unstructured-mesh support via MPASdiag (no convert-mpas needed)
* NaN-aware FSS pipeline to handle missing data robustly
* Per-experiment CSV extraction and batch plotting for comprehensive analysis
* Modular design with clear separation of concerns for easy maintenance and extension
* Comprehensive documentation and unit tests to ensure reliability and ease of use
* Open-source MIT license for broad accessibility and collaboration

## Quick start

```bash
# install Git LFS if not already installed (required for large test datasets)
git lfs install

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

# YAML configuration

MODvx uses flat YAML configuration files to specify all aspects of the verification workflow. All parameters live at the top level (no nested sections), making configs easy to read, diff, and override from the CLI. Copy `configs/default.yaml` as your starting point and edit the sections below.

```yaml
# configs/default.yaml  —  annotated reference configuration

# ---- experiment / time range -------------------------------------------

experiment_name: "liuz_coldstart_15km2025"  # used in output filenames and plot titles
initial_cycle_start: "20250613T00"           # first cycle (yyyymmddThh)
final_cycle_start:   "20250709T00"           # last cycle  (yyyymmddThh)
forecast_step_hours: 12        # interval between forecast output files (hours)
observation_interval_hours: 1  # accumulation window of the observation dataset (hours)
cycle_interval_hours: 24       # interval between initialisation cycles (hours)
forecast_length_hours: 48      # total lead time to verify (hours)
precip_accum_hours: 0          # hours over which to accumulate precip; set to 0 to disable accumulation and use raw fields

# ---- MPAS mesh settings (leave mpas_grid_file empty for regular-grid input) ----

mpas_grid_file: ""      # path to MPAS static grid file; empty = not an MPAS run
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
  SINGV:     "singv_domain_mask.nc"
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

base_dir:   "."           # root directory; all other paths resolved against this
fcst_dir:   "fcst"        # forecast NetCDF files
obs_dir:    "obs/FIMERG"  # observation NetCDF files
mask_dir:   "masks"       # region mask NetCDF files
output_dir: "output"      # FSS result NetCDF files
csv_dir:    "fss_csv"     # extracted CSV tables
plot_dir:   "fss_plots"   # generated plots
log_dir:    "logs"        # run logs (when enable_logs: true)
debug_dir:  "debug"       # intermediate debug outputs

# ---- filename templates -------------------------------------------------

observation_template: >-
  {obs_dir}/IMERG.A01H.VLD{date_key}.S{date_key}T000000.E{date_key}T235959.{vintage}.V07B.SRCHHR.X360Y180.R1p0.FMT.nc

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
| `target_resolution` | Use `"obs"` for most runs; `"fcst"` when the forecast grid is coarser than the obs. |
| `mpas_grid_file` | Set this to the MPAS static grid NetCDF when running with native MPAS diag files (no convert-mpas required). |
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
│   ├── parallel.py          # MPI wrapper
│   ├── file_manager.py      # All file I/O
│   ├── mpas_reader.py       # Native MPAS mesh reader (via MPASdiag)
│   ├── data_validator.py    # Grid prep pipeline
│   ├── perf_metrics.py      # FSS + contingency-table metrics (POD, FAR, CSI, FBIAS, ETS)
│   └── visualizer.py        # Plotting
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
