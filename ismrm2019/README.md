## ISMRM 2019

This material was presented at the ISMRM Workshop 2019.

**Title**: BART Reconstruction Toolbox / Iterative Reconstruction Toolbox in Python  
**Meeting**: [2019 ISMRM Annual Meeting, Montreal, Canada](https://www.ismrm.org/19m/)  
**Date**: May 12, 2019


### Demo dependencies

The workshop material was tested with [BART version 0.4.04](https://github.com/mrirecon/bart/releases/tag/v0.4.04).

The demos use Jupyter notebooks with Python and Bash kernels. This is easiest to accomplish with Anaconda.
After installing Anaconda, install the dependencies:
```bash
pip install h5py numpy matplotlib bash_kernel 
```

Finish installing the Bash kernel with the command,
```bash
python -m bash_kernel.install
```

## Schedule
This folder contains standalone demos that show different BART use cases. The demos are self-documented within
each directory, and are summarized below. Each tutorial is set up as jupyter notebook and can interactively be tested using Binder.

- **Tutorial 1**: Introduction to the BART command-line tools
  - [Jupyter Notebook](./intro/intro.ipynb)
  - [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-workshop/master?filepath=ismrm2019/intro/intro.ipynb)
- **Tutorial 2**: Compute g-factor using Python and BART
  - [Jupyter Notebook](./gfactor-demo/gfactor-demo-real_data.ipynb)
  - [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-workshop/master?filepath=ismrm2019/gfactor-demo/gfactor-demo-real_data.ipynb)
- **Tutorial 3**: Build a non-Cartesian SENSE reconstruction tool with the BART C API
  - [Jupyter Notebook](./sense-recon/sense-recon.ipynb)
  - [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrirecon/bart-workshop/master?filepath=ismrm2019/sense-recon/sense-recon.ipynb)