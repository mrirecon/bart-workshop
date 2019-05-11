# BART Workshop Materials

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mikgroup/bart-workshop/master?filepath=setup.ipynb)

This repository contains information and demos for the [Berkeley Advanced Reconstruction Toolbox (BART)](http://mrirecon.github.io/bart).
This material will be presented at the [2019 ISMRM Annual Meeting, Montreal, Canada](https://www.ismrm.org/19m/). Previously, the material was presented
at the [2016 ISMRM Workshop on Data Sampling & Image Reconstruction](http://www.ismrm.org/workshops/Data16/).

## **NEW** Run demos in the browser through MyBinder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mikgroup/bart-workshop/master?filepath=setup.ipynb)  
[Click the icon](https://mybinder.org/v2/gh/mikgroup/bart-workshop/master?filepath=setup.ipynb)



## Purpose
The purpose of this repository is to host and share demos and workshop materials for BART. From the [website](http://mrirecon.github.io/bart):

> The Berkeley Advanced Reconstruction Toolbox (BART) is a free and open-source image-reconstruction framework
> for Computational Magnetic Resonance Imaging. It consists of a programming library and a toolbox of command-line
> programs. The library provides common operations on multi-dimensional arrays, Fourier and wavelet transforms,
> as well as generic implementations of iterative optimization algorithms. The command-line tools provide direct
> access to basic operations on multi-dimensional arrays as well as efficient implementations of many calibration
> and reconstruction algorithms for parallel imaging and compressed sensing.

## Getting Started
The most up-to-date information can be found at the official BART website: http://mrirecon.github.io/bart.

The workshop material was tested with [BART version 0.4.04](https://github.com/mrirecon/bart/releases/tag/v0.4.04)

### Download
The source code is available at [https://github.com/mrirecon/bart/archive/v0.4.04.tar.gz](https://github.com/mrirecon/bart/archive/v0.4.04.tar.gz).
Untar and navigate to the bart directory:
```bash
wget https://github.com/mrirecon/bart/archive/v0.4.04.tar.gz
tar -xvvf v0.4.04.tar.gz && mv bart-0.4.04 bart
cd bart
```

### Quick Installation
See the [Quick-Install guide](doc/quick-install.md) for quick installation instructions.


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
The [`demos`](demos) directory contains standalone demos that show different BART use cases. The demos are self-documented within
each directory, and are summarized below.

1. Introduction to the BART command-line tools ([`intro`](demos/intro/intro.ipynb))
1. Compute g-factor using Python and BART ([`gfactor`](demos/gfactor-demo/gfactor-demo-real_data.ipynb))
1. Build a non-Cartesian SENSE reconstruction tool with the BART C API ([`sense-recon`](demos/sense-recon/sense-recon.ipynb))

#### Additional demos:
1. Simulate phantom data and compare regularized reconstructions  ([`pics-phantom`](demos/pics-phantom))
1. Reconstruct an axial slice of dynamic contrast enhanced (DCE) data ([`pics-dce`](demos/pics-dce))
1. Build a GRASP reconstruction tool with bash scripting and BART command-line tools ([`grasp`](demos/grasp))
1. Use the Wave-CS reconstruction tool using the BART Matlab API ([`wave`](demos/wave-cs))

