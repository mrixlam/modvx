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
conda install -y -c conda-forge numpy scipy xarray matplotlib pyyaml

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
modvx run -c configs/bench_liuz.yaml --vxdomain GLOBAL --backend multiprocessing --nprocs 12

# Run with native MPAS mesh input (no convert-mpas needed)
modvx run -c configs/mri_test.yaml --vxdomain GLOBAL

# Extract results to CSV
modvx extract-csv -c configs/mri_test.yaml

# Generate plots for ALL metrics (fss, pod, far, csi, fbias, ets)
modvx plot --all -c configs/mri_test.yaml

# Generate plots for specific metrics only
modvx plot --all -c configs/mri_test.yaml --metric fss,pod,csi

# Single plot for a specific metric, domain, threshold, window
modvx plot -c configs/mri_test.yaml --domain GLOBAL --thresh 90 --window 3 --metric ets

# List available options
modvx validate -c configs/mri_test.yaml
```

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

## License

MIT — see [pyproject.toml](pyproject.toml) for details.
