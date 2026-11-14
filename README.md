# Fitting the Jansen-Rit Model to Rat ERPs

Inference of Jansen-Rit neural mass model parameters from rat EEG evoked responses recorded during wakefulness and under anesthesia, using two independent approaches: MCMC (NumPyro/NUTS) and simulation based inference (SBI).

This was an EPIC excellence programme project (MSc Physics & Astronomy, Ghent University) under Prof. Daniele Marinazzo, Dept. of Data Analysis. The same material is also available as an EBRAINS Collab:
https://wiki.ebrains.eu/bin/view/Collabs/fitting-the-jansen-rit-model-to-rat-erps

## What is in here

Three notebooks, to be run in this order:

| Notebook | What it does |
|---|---|
| `01_general.ipynb` | Preprocessing of the raw recordings, overview of the dataset, Fourier spectra, HF/LF power ratio, sliding window PCIst, Jansen-Rit background and simulations, PCIst explanation. Saves the preprocessed data. |
| `02_mcmc.ipynb` | NUTS fit of the Jansen-Rit model directly to a single subject's mean ERP waveform, including the prior attempts, diagnostics and sweeps that led to the final setup. |
| `03_sbi.ipynb` | SNPE based inference of the gain parameters from PCIst summary statistics, through five pipelines with increasing conditioning, ending with a gain matrix similarity comparison. |

Shared code the notebooks import:

```
data.py
jansen_rit.py
mcmc.py
pcist.py
inference_utils.py
plotting.py
```

Keep the notebooks and the `.py` files together in one folder. The notebooks import the modules directly (for example `import data as dat`, `import pcist`), so no extra path setup is needed.

`01_general.ipynb` is the only notebook that reads the raw `.mat` files. It saves the preprocessed data to `data/processed/`, and `02_mcmc.ipynb` and `03_sbi.ipynb` load from there. So run `01_general.ipynb` first.

For now both MCMC and SBI infer only the four gain parameters (`g_1` to `g_4`), with the other Jansen-Rit parameters fixed at their defaults. The parameter names and priors are defined from lists, so extending to more parameters later means extending those lists.

## Data

The raw data is not in this repository. It is a rat EEG dataset published on EBRAINS:

https://search.kg.ebrains.eu/instances/9400031b-dc13-4444-b161-c574d184fdca

A stimulus is delivered at t = 0 and the evoked response is recorded on 16 electrodes, both awake and under one of three anesthetics (ketamine, propofol, sevoflurane). Not every rat went through every anesthesia session.

After downloading, put the `.mat` files in a `data/` folder next to the notebooks, in one subfolder per anesthesia, one file per subject and condition. This is the layout `01_general.ipynb` expects:

```
data/PCI-like_keta1/PCI-like_j12_keta1-r3.mat
data/PCI-like_keta1/PCI-like_j13_wake-r2.mat
data/PCI-like_prop1/PCI-like_j12_prop1-r1.mat
data/PCI-like_prop1/PCI-like_j13_wake-r1.mat
data/PCI-like_sevo1/PCI-like_j12_sevo1-r2.mat
data/PCI-like_sevo1/PCI-like_j12_wake-r2.mat
```

Wake recordings are duplicated across the anesthesia subfolders (one wake file per subject per anesthesia session), they are not stored separately.

## Environment

Developed on WSL2 (Ubuntu) on Windows with an NVIDIA RTX 3050 Laptop GPU (4GB VRAM). It should also run on CPU, just slower. Batch sizes in the simulation, PCIst and SBI code are kept modest because of the small GPU.

Python 3.10.19 with these versions:

```
numpy==2.2.6
scipy==1.15.3
xarray==2025.6.1
pandas==2.3.3
scikit-learn==1.7.2
matplotlib==3.10.8
seaborn==0.13.2
statannotations==0.7.2
arviz==0.23.4
jax==0.6.2
jaxlib==0.6.2          (GPU build)
torch==2.5.1+cu124      (GPU build)
sbi==0.25.0
```

Also needed, but not pinned to a version during development:

```
numpyro       (02_mcmc.ipynb)
netCDF4       (xarray needs it to read and write the .nc files)
ipywidgets
tqdm
```

`01_general.ipynb` optionally cross checks the JAX PCIst implementation in `pcist.py` against the original numpy implementation. That cell is skipped if the package is missing and nothing downstream needs it:

```
pip install git+https://github.com/renzocom/PCIst.git
```

On a different jax/torch/CUDA combination both should still work on CPU. GPU support depends on installing jaxlib and torch builds that match your CUDA version.

## Using this code

You are free to use this code, reuse parts of it, or build further on it, as long as you credit me, Milan Christiaens, as the original author. The code is released under the MIT license, see `LICENSE`.

## Acknowledgments

- Prof. Daniele Marinazzo, Dept. of Data Analysis, Ghent University, supervisor of the project
- Amirhossein Esmaeili, who wrote the JAX implementation of PCIst (`pcist.py`) and originally came up with the forward projection idea (gain matrices, going from a single channel simulation to the 16 channel space), which I then heavily refined
- Comolatti et al., for the PCIst algorithm and the numpy reference implementation (https://github.com/renzocom/PCIst)
- The EBRAINS rat EEG dataset linked above
- Jansen-Rit model in O. David's ERP formulation, as used in DCM
