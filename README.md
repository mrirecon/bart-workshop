# BART Workshop Materials

This repository contains information and demos for the [Berkeley Advanced Reconstruction Toolbox (BART)](http://mrirecon.github.io/bart).
This material was presented at the [2016 ISMRM Workshop on Data Sampling & Image Reconstruction](http://www.ismrm.org/workshops/Data16/).

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

The workshop material was tested with [BART version 0.3.01](https://github.com/mrirecon/bart/releases/tag/v0.3.01)

### Download
The source code is available at [https://github.com/mrirecon/bart/archive/v0.3.01.tar.gz](https://github.com/mrirecon/bart/archive/v0.3.01.tar.gz).
Untar and navigate to the bart directory:
```bash
wget https://github.com/mrirecon/bart/archive/v0.3.01.tar.gz
tar -xvvf v0.3.01.tar.gz && mv bart-0.3.01 bart
cd bart
```

### Quick Installation
See the [Quick-Install guide](doc/quick-install.md) for quick installation instructions.



## Demos
The [`demos`](demos) directory contains standalone demos that show different BART use cases. The demos are self-documented within
each directory, and are summarized below.

1. Simulate phantom data and compare regularized reconstructions ([`pics-phantom`](demos/pics-phantom))
1. Reconstruct an axial slice of dynamic contrast enhanced (DCE) data ([`pics-dce`](demos/pics-dce))
1. Build a GRASP reconstruction tool with bash scripting and BART command-line tools ([`grasp`](demos/grasp))
1. Build a Wave-CS reconstruction tool in C using the BART C API ([`wave`](demos/wave-cs))
1. Simulate multi-channel data and computer g-factor using Python and BART([`gfactor`](demos/gfactor-demo))

