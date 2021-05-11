## ISMRM 2016

This material was presented at the ISMRM Workshop 2019.

**Title**: BART Reconstruction Toolbox / Iterative Reconstruction Toolbox in Python  
**Meeting**: [2019 ISMRM Annual Meeting, Montreal, Canada](https://www.ismrm.org/19m/)  
**Date**: May 12, 2019


### Demo dependencies
The demos use Jupyter notebooks with Python and Bash kernels. This is easiest to accomplish with Anaconda.
After installing Anaconda, install the dependencies:
```bash
pip install h5py numpy matplotlib bash_kernel 
```

Finish installing the Bash kernel with the command,
```bash
python -m bash_kernel.install
```

## Demos
This folder contains standalone demos that show different BART use cases. The demos are self-documented within
each directory, and are summarized below.

1. Introduction to the BART command-line tools ([`intro`](intro/intro.ipynb))
2. Compute g-factor using Python and BART ([`gfactor`](gfactor-demo/gfactor-demo-real_data.ipynb))
3. Build a non-Cartesian SENSE reconstruction tool with the BART C API ([`sense-recon`](sense-recon/sense-recon.ipynb))
