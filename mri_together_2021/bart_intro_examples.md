---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="MIGYtH14DwXD" -->
# A Hands-On Introduction to BART

[Martin Uecker](mailto:uecker@tugraz.at)$^{\dagger,*,+}$, [Nick Scholand](mailto:scholand@tugraz.at)$^{*,+}$, [Moritz Blumenthal](mailto:moritz.blumenthal@med.uni-goettingen.de)$^*$, [Xiaoqing Wang](mailto:xiaoqing.wang@med.uni-goettingen.de)$^{*,+}$

$^{\dagger}$Graz University of Technology, $^*$University Medical Center Göttingen, $^+$German Centre for Cardiovascular Research, Partner Site Göttingen
<!-- #endregion -->

<!-- #region id="hBjVg_bNnHzj" -->
## Requirements



<!-- #endregion -->

<!-- #region id="e5RxFHCDnKGt" -->
### Local Usage
- Install bart from its [github repository](https://github.com/mrirecon/bart)  (the newest version is needed for machine learning)
- Set the `TOOLBOX_PATH` to the BART directory and add it to the `PATH`

```bash
export TOOLBOX_PATH=/path/to/bart  
export PATH=$TOOLBOX_PATH:$PATH
```

Although the simplest way to call the BART CLI tools is from a terminal, there are also wrapper functions that allow the tools to be used from Matlab and Python. These are located under the `$TOOLBOX_PATH/matlab` and `$TOOLBOX_PATH/python` directories.
<!-- #endregion -->

<!-- #region id="f09HrbbxDwXJ" -->
### Online Usage
MyBinder and Google Colaboratory can be used to access a Jupyter instance with BART with a browser. In the following we install and configure BART for both.
<!-- #endregion -->

<!-- #region id="MZ-L3VADnvy2" -->
#### Check for GPU

**Google Colaboratory** provides access to a GPU. To enable it:

- Go to Edit → Notebook Settings
- choose GPU from Hardware Accelerator drop-down menu

**MyBinder** does not provide GPU access.

The following code will automatically detect which service you are using.
<!-- #endregion -->

```python id="pqkS51t1DwXK"
# Check if notebook runs on colab
import sys, os

os.environ['COLAB'] = 'true' if ('google.colab' in sys.modules) else 'false'

# FIXME: Colab without GPU not supported yet
os.environ['CUDA'] = '1' if ('google.colab' in sys.modules) else '0'
```

```bash id="u5ylS2IpDwXM" colab={"base_uri": "https://localhost:8080/"} outputId="3df17531-c48f-4ce4-c51f-04b96cd176cb"

# Prepare GPUs if on Google Colab
if $COLAB; then

    # Use CUDA 10.1 when on Tesla K80

    # Determine GPU Type
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

    echo "GPU Type:"
    echo $GPU_NAME

    if [ "Tesla K80" = "$GPU_NAME" ]; then

        echo "GPU type Tesla K80 does not support CUDA 11. Set CUDA to version 10.1."

        cd /usr/local
        rm cuda
        ln -s cuda-10.1 cuda
    fi

    echo "GPU Information:"
    nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv
    nvcc --version
fi
```

<!-- #region id="leAhIZJ9oWWR" -->
#### BART Installation

Here we install BARTs dependencies, clone its repository from github, and compile it.
<!-- #endregion -->

```bash id="KYpxLsEEDwXN"

# MyBinder has BART already installed via the container
if $COLAB; then

  # Install BARTs dependencies
  apt-get install -y make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev &> /dev/null

  # Clone Bart
  [ -d /content/bart ] && rm -r /content/bart
  git clone https://github.com/mrirecon/bart/ bart &> /dev/null
fi
```

```bash id="tDWAtWn0DwXP" colab={"base_uri": "https://localhost:8080/"} outputId="31f7a9eb-ffff-4dfc-87f8-29978f0b99d9"

if $COLAB; then

  cd bart

  # Configuration
  COMPILE_SPECS=" PARALLEL=1
                  CUDA=$CUDA
                  CUDA_BASE=/usr/local/cuda
                  CUDA_LIB=lib64
                  OPENBLAS=1
                  BLAS_THREADSAFE=1"

  printf "%s\n" $COMPILE_SPECS > Makefiles/Makefile.local

  # Compile BART
  make &> /dev/null && echo ok
fi
```

<!-- #region id="hb0N6uqDDwXR" -->
#### Setup Environment for BART

After downloading and compiling BART, the next step simplifies the handling of BARTs command line interface inside an ipyhton jupyter notebook. We add the BART directory to the PATH variable and include the python wrapper for reading *.cfl files:
<!-- #endregion -->

```python id="sU6XPDEyDwXU"
os.environ['TOOLBOX_PATH'] = "./bart"
os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
sys.path.append(os.environ['TOOLBOX_PATH'] + "/python/")
```

<!-- #region id="skXvcEL4DwXW" -->
Check BART setup:
<!-- #endregion -->

```bash id="SQAZz46ODwXW" colab={"base_uri": "https://localhost:8080/"} outputId="c21295b0-3c25-412c-bbc7-8f1e4b66e70e"
echo "# BART version: "
bart version
```

<!-- #region id="-B54--FpFnPB" -->
### Setup Visualization Helper

For this tutorial we will visualize some images. Therefore, we need a helper function and some python libraries.

<!-- #endregion -->

```python id="jjk7WRl5FqTh"
# More python libraries
import cfl
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import Image

def plot_map(dataname, colorbar, vmin, vmax, cbar_label):

    # Import data        
    data = np.abs(cfl.readcfl(dataname).squeeze())

    # Import colorbar type
    colorbartype =  colorbar

    # Set zero to a black color for a masking effect
    my_cmap = cm.get_cmap(colorbartype, 256)
    my_cmap.set_bad('black')

    data = np.ma.masked_equal(data, 0)

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(data, interpolation='nearest', cmap=my_cmap, vmin=vmin, vmax=vmax)

    # Style settings
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params()

    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    ax1.set_axis_off()

    plt.show()
```

<!-- #region id="SdvGf7xsjhpb" -->
### Download Supporting Materials
For this tutorial, we also need several supporting materials (figures, plotting scripts and compressed data for ML part). They are stored in the GitHub repository and need to be downloaded.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="ouqzdOLhjp1t" outputId="2b69c4e3-32bb-488a-8e2c-8222be0cac42"

# Download the required supporting material if it is not already there
[ ! -f data.zip ] && wget -q https://github.com/mrirecon/bart-workshop/raw/master/mri_together_2021/data.zip
unzip -n data.zip

# Download Raw Data for IR FLASH parts
name=IR-FLASH

if [[ ! -f ${name} ]]; then
  echo Downloading ${name}
  wget -q https://zenodo.org/record/4060287/files/${name}.cfl
  wget -q https://zenodo.org/record/4060287/files/${name}.hdr

  mv ${name}.cfl data/${name}.cfl
  mv ${name}.hdr data/${name}.hdr
fi

head -n2 data/${name}.hdr
```

<!-- #region id="5Gqo3SJqDwXZ" -->
## BART Fundamentals
BART provides a number of tools for MRI image reconstruction and multi-dimensional array manipulation.
<!-- #endregion -->

```python id="dM-uW8dEDwXZ" colab={"base_uri": "https://localhost:8080/"} outputId="3e206ee6-2140-4f9b-bed7-9c52e7ceaed8"
# get a list of BART commands by running bart with no arguments:
! bart
```

<!-- #region id="_zfuPZNkDwXb" -->
### BART Command Structure

The command structure follows

> **`bart`** + **`command`** + **`options`** + **`input / output`**

Each BART command consists of a number of optional arguments, followed by input and output files.To get some information about a command, use the -h flag. Optional arguments are indicated by brackets, and files are indicated by carrot symbols.

<!-- #endregion -->

```python id="mdwlVvNMDwXe" colab={"base_uri": "https://localhost:8080/"} outputId="958ea2b7-6787-4c08-dde5-008fa6fa294c"
! bart phantom -h
```

<!-- #region id="l65K0lkLDwXe" -->
The `phantom` tool includes the option `-k` to create it in k-space, and `-x` to specify the size.
<!-- #endregion -->

```python id="G4IikpLXDwXe" colab={"base_uri": "https://localhost:8080/", "height": 852} outputId="ac8ce1f0-1fd8-4b2d-bd87-732a3669d9d8"
# Create Cartesian k-space phantom (256 samples)
! bart phantom -x 256 -k shepp_logan

! echo "Shepp-Logan k-space phantom"
plot_map("shepp_logan", "viridis", 0, 0.02, '')
```

<!-- #region id="SnC5NmoMQPFb" -->

<!-- #endregion -->

<!-- #region id="WJzGVMeuDwXe" -->
### Data File Format
All data files are multi-dimensional arrays. By running the `phantom` command, we made new files on disk, with the names  
`shepp_logan.cfl` and `shepp_logan.hdr`

The header (`hdr`) is a text file that contains the data dimensions and additional information. The data file (`cfl`) contains the complex single-precision raw data in column-major order and with interleaved real and imaginary parts.  

The header file is a raw text file and can be opened with any text editor. The complex-float file is a binary file. Matlab and Python readers/writers are available under the `matlab` and `python` directories, respectively.

### View data dimensions
Because the header file is a text file, we can directly print it:
<!-- #endregion -->

```python id="DWONQd_zDwXf" colab={"base_uri": "https://localhost:8080/"} outputId="18bd6188-1a21-485e-90b2-2e1d56c82c94"
! cat shepp_logan.hdr
```

<!-- #region id="741w3qC8DwXf" -->
Although not discussed here, BART can write to other file formats that might not have a text header. Thus, it is better to use the show command:
<!-- #endregion -->

```python id="GQFw_AUGDwXh" colab={"base_uri": "https://localhost:8080/"} outputId="b6c1baf0-7dea-4df5-f02b-d2dd3c11d6f1"
! bart show -m shepp_logan
```

<!-- #region id="jzz1q1nUDwXh" -->
Our dataset is 16-dimensional, but only the first two dimensions are non-singleton.

By convention, the dimensions are `[X, Y, Z, C, M, T, F, ...]`,
where `(X, Y, Z)` are the spatial matrix dimensions,  
`C` and `M` are the coil dimensions and ESPIRiT maps dimensions, respectively,  
`T` and `F` are used for echo times and coefficient dimensions,   
followed by other higher order dimensions such as flow encoding, etc.
<!-- #endregion -->

<!-- #region id="rkEKZc2wDwXh" -->
### Using Bitmasks to select dimensions
Let's reconstruct our k-space phantom using a inverse Fast Fourier Transform (iFFT).
<!-- #endregion -->

```python id="7IinGE42DwXi" colab={"base_uri": "https://localhost:8080/", "height": 852} outputId="900b780f-c58e-4067-e8be-7477bdfcad19"
# Perform FFT reconstruction
! bart fft -u -i 3 shepp_logan shepp_logan_rec

! echo "IFFT of Shepp-Logan phantom"
plot_map("shepp_logan_rec", "viridis", 0, 0.005, '')
```

```python id="R7S8Vs6QDwXk" colab={"base_uri": "https://localhost:8080/"} outputId="3d16fd8c-c9e9-4d98-ffdc-6dcde4a5f806"
# Show help for fft command
! bart fft -h
```

<!-- #region id="E8Ja88-DDwXk" -->
Thus, we performed an inverse (`-i`) unitary (`-u`) Fast Fourier Transform on the image dimensions **`(0, 1)`** specified by the bitmask **`3`**.

<!-- #endregion -->

<!-- #region id="Q66iAkUMDwXk" -->
BART loops over dimensions selected by *bitmasks*. This is a powerful approach for perfoming multi-dimensional operations, as most tools will work on arbitrarily chosen dimensions.  

In our case, we wanted to perform an iFFT along dimensions 0 and 1, and the corresponding bitmask is calculated as:  
<center>
$ \text{bitmask}=2^{~0} + 2^{~1} = 3$
</center> <br>
BART also provides a command-line tool to calculate the bitmasks for specific dimensions.
<!-- #endregion -->

```python id="w_Lgk8REDwXl" colab={"base_uri": "https://localhost:8080/"} outputId="fb143293-7f41-45a2-a623-3df4b647a297"
# Calculate bitmask for active dimensions 0 and 1
! bart bitmask 0 1
```

<!-- #region id="tIKCcmsTDwXl" -->
## BART Examples

<!-- #endregion -->

<!-- #region id="ePpy5MiNtR7d" -->
### Subspace T1 Mapping

A complete tutorial for subspace T1 mapping with BART can be found in the [3rd BART Webinar Materials](https://github.com/mrirecon/bart-webinars/tree/master/webinar3).
<!-- #endregion -->

<!-- #region id="sDOZbCGHDwXm" -->
#### Theory

**Single-Shot Inversion-Prepared T1 Mapping**

<img src="https://github.com/mrirecon/bart-workshop/blob/master/ismrm2021/model_based/IR_FLASH.png?raw=1" style="width: 550px;">






<!-- #endregion -->

<!-- #region id="xUbjrToOuTd7" -->
#### Dictionary Generation, SVD and Temporal Basis

Calculate dictionary.
<!-- #endregion -->

```bash id="6fDErOkBuOrv"

TR=0.0041

# Dictionary characteristics
## R1s
#NUM_R1S=1000
# only use 100 for demo
NUM_R1S=100
MIN_R1S=5e-3
MAX_R1S=5

## Mss
NUM_MSS=100
MIN_MSS=1e-2
MAX_MSS=1

# Read file dimensions from downloaded dataset
REP=`bart show -d 10 data/IR-FLASH`

# Simulate dictionary based on the `signal` tool
bart signal -F -I -n$REP -r$TR \
            -1 $MIN_R1S:$MAX_R1S:$NUM_R1S \
            -3 $MIN_MSS:$MAX_MSS:$NUM_MSS  dicc

# reshape the dicc 6th and 7th dimension to have all the elements 
# concentrated in the 6th dimension
bart reshape $(bart bitmask 6 7) $((NUM_R1S * NUM_MSS)) 1 dicc dicc_reshape

# squeeze the reshaped dictionary to remove singleton dimensions
bart squeeze dicc_reshape dicc_squeeze
```

<!-- #region id="SGYd3Hvo5WCp" -->
Perform a SVD to create our temporal basis with a given number of coefficients.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="GsBB2Hv05VYl" outputId="d1480cd4-cfdb-4029-8712-7e71dda0c40b"

NUM_COE=4

# Perform an SVD of the dictionary and output a
# decomposition of the resulting matrix
bart svd -e dicc_squeeze U S V

# Extract desired number of orthonormal columns from U
bart extract 1 0 $NUM_COE U basis0

# Transpose the basis to have time in the 5th dimension 
# and coefficients in the 6th dimension
bart transpose 1 6 basis0 basis1
bart transpose 0 5 basis1 basis

# Print the transposed basis dimensions
echo "Temporal Basis"
head -n2 basis.hdr
```

<!-- #region id="KkLHKkmhukOC" -->
#### Coil Compression
To reduce the size of our dataset and therefore also decrease the computational complexity, we perform a coil compression with the `cc` command. By passing `-A` we choose to use all possible data and to reduce the dataset to 8 virtual coils with `-p`.

<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="H7m5VK89uqqn" outputId="178d2c5b-9524-49f7-8e30-6d7b9492a987"

# Transpose the 0th and 1st dimension of the downloaded data
# to ensure compatibility with BARTs non-Cartesian tools
bart transpose 0 1 data/IR-FLASH ksp

# Perform coil compression
bart cc -A -p 8 ksp ksp_cc
```

<!-- #region id="_JUtkpkCu2XL" -->
#### Trajectory Generation

In the next step we generate a trajectory with the `traj` tool. To match the acquisition of the downloaded data, we need to specify radial `-r`, centered `-c`, double-angle `-D`, 7th tiny golden-angle `-G -s7` sampling. The number of turns are passed using `-t`, the spokes using `-y` and the samples using `-x`.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="C5LkG-ZPvFl0" outputId="2b31b36b-3d1c-45e3-f2c7-b57888b36c6f"

# Tiny golden angle
NUM_TGA=7

# Read file dimensions from downloaded dataset
READ=`bart show -d 0 data/IR-FLASH`
SPOKES=`bart show -d 2 data/IR-FLASH`
REP=`bart show -d 10 data/IR-FLASH`

# Create the trajectory using the `traj` tool
bart traj -r -c -D -G -x$READ -y$SPOKES -s$NUM_TGA -t$REP traj

# Print out its dimensions
echo "Trajectory"
head -n2 traj.hdr
```

<!-- #region id="sp04vChGvejm" -->
#### Gradient Delay Correction
Because the signal is following an IR FLASH curve, the gradient-delay correction should be applied to the data from the last repetitions which are in a steady-state. Therefore, we extract repetitions from the end of the trajectory and the dataset using the `extract` command. Have in mind that the time dimension is the 10th here!
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="eiCjJxmzvsPl" outputId="3d68a7db-ebde-4204-ecee-0df26356ef02"

# Tiny golden angle
NUM_TGA=7

# Define the number of TRs the gradient delay correction should be 
# performed on (Steady-State)
FRAMES_FOR_GDC=100

# Read file dimensions from downloaded dataset
READ=`bart show -d 0 data/IR-FLASH`
SPOKES=`bart show -d 2 data/IR-FLASH`
REP=`bart show -d 10 data/IR-FLASH`

# Extract the DATA_GDC last time frames from the trajectory and the dataset
bart extract 10 $((REP - FRAMES_FOR_GDC)) $REP traj traj_extract
bart extract 10 $((REP - FRAMES_FOR_GDC)) $REP ksp_cc ksp_extract

# Transpose the 2nd and 10th dimension for later use with the `estdelay` tool
bart transpose 10 2 traj_extract traj_extract1
bart transpose 10 2 ksp_extract ksp_extract1

# Estimate and store the gradient delays usign RING
GDELAY=$(bart estdelay -R traj_extract1 ksp_extract1)

echo "Gradient Delays: "$GDELAY

# Calculate the trajectory with known gradient delays
bart traj -r -c -D -G -x$READ -y$SPOKES -s$NUM_TGA -q$GDELAY -t$REP trajn2

# 2x oversampling
bart scale 0.5 trajn2 trajn
```

<!-- #region id="vMR9HvyLv7RN" -->
#### Coil Sensitivity Estimation

The coil profile estimation is similar to the gradient delay estimation performed on some of the last timesteps of the IR FLASH dataset.`extract` command.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="Rw2nBqkev-dV" outputId="7e080853-6744-4e7f-8549-25b747e7b899"

# number of TRs at end of the dataset to be used
FRAMES_FOR_CSE=300

# Read file dimensions from downloaded dataset
READ=`bart show -d 0 data/IR-FLASH`
SPOKES=`bart show -d 2 data/IR-FLASH`
REP=`bart show -d 10 data/IR-FLASH`

# Extract last time frames (10th dim) from trajectory and k-space data
bart extract 10 $((REP - FRAMES_FOR_CSE)) $REP trajn traj_ss
bart extract 10 $((REP - FRAMES_FOR_CSE)) $REP ksp_cc ksp_cc_ss

bart transpose 2 10 traj_ss traj_ss2
bart transpose 2 10 ksp_cc_ss ksp_cc_ss2

# Apply an inverse non-uniform FFT
bart nufft -i -d$(($READ / 2)):$(($READ / 2)):1 traj_ss2 ksp_cc_ss2 img


# transform reconstruction in image space back to k-space
bart fft -u $(bart bitmask 0 1 2) img ksp_grid

# Estimate coil sensitivities
bart ecalib -S -t0.01 -m1 ksp_grid sens_invivo

cat sens_invivo.hdr
```

```bash id="aBq-9xhMMX0-"
# Reshape and flip coefficient maps for improved visualization

# Read file dimensions
READ=`bart show -d 0 sens_invivo`
COILS=`bart show -d 3 sens_invivo`

## Merge all coefficients in the column dimension (1st/phase1)
bart reshape $(bart bitmask 1 3) $((READ * COILS)) 1 sens_invivo sens_invivo_lin

## Flip the map in row dimension
bart flip $(bart bitmask 0) sens_invivo_lin sens_invivo_flip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 157} outputId="a441ad4b-b4f1-4048-c1f8-339a2a553011" id="CyEMOkmIfGOp"
! echo "In-Vivo Sensitivity Maps"
plot_map("sens_invivo_flip", "viridis", 0, 1, '')
```

<!-- #region id="3AxOykoMwTps" -->
#### Subspace-Constrained Reconstruction

<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="N8zOL3vpwZOy" outputId="aac108b0-a474-4a5d-e9a9-5938f6836a44"

# Transpose dimensions for working with PICS tool
bart transpose 5 10 trajn traj_final
bart transpose 5 10 ksp_cc ksp_final

READ=`bart show -d 0 sens_invivo`

DEBUG=4
ITER=100
REG=0.0015

if $COLAB; then
    GPU=-g;
else
    GPU=''
fi

bart pics   $GPU -e -d $DEBUG -i$ITER \
            -RW:$(bart bitmask 0 1):$(bart bitmask 6):$REG \
            -t traj_final -B basis \
            ksp_final sens_invivo coeff_maps

echo "Reconstructed Coefficients"
head -n2 coeff_maps.hdr
```

<!-- #region id="y_rpDr3kwnRL" -->
#### Visualization of Reconstructed Maps
<!-- #endregion -->

```bash id="mcU_b2E8wstT"
# Reshape and flip coefficient maps for improved visualization

READ=`bart show -d 0 coeff_maps`
NCOE=`bart show -d 6 coeff_maps`

## Merge all coefficients in the column dimension (1st/phase1)
bart reshape $(bart bitmask 1 6) $((READ * NCOE)) 1 coeff_maps subspace_maps

## Flip the map in row dimension to have correct orientation
bart flip $(bart bitmask 0) subspace_maps subspace_maps1
```

```python colab={"base_uri": "https://localhost:8080/", "height": 256} id="aB2llFy0hf2i" outputId="19e93010-d4c3-47f8-cef8-4235b36ae17b"
! echo "Subspace Coefficient Maps"
plot_map("subspace_maps1", "viridis", 0, 0.5, '')
```

<!-- #region id="_-JV1PCDpbvW" -->
### Model-Based T1 Mapping

A complete tutorial for model-based reconstructions in BART can be found in the [Workshop Material of the ISMRM 2021](https://github.com/mrirecon/bart-workshop/tree/master/ismrm2021).
<!-- #endregion -->

<!-- #region id="BC-8Qb7uqG8-" -->
**Operator chain of parallel imaging and signal model (nonlinear)**

<img src="https://github.com/mrirecon/bart-workshop/blob/master/ismrm2021/model_based/operator_chain.png?raw=1" style="width: 400px;">

$$F: x \mapsto y = {\mathcal{P} \mathcal{F} C} \cdot {M(x_{p})}$$
- $\mathcal{P}$ - sampling pattern
- $\mathcal{F}$ - Fourier transform
- $C$ - coil sensitivity maps
- $M(\cdot)$ - MR physics model
- $x_{p}$ - MR parameters
- $y$ - acquired kspace data
<!-- #endregion -->

<!-- #region id="SaIAOQw4p5LI" -->
#### Optimization

We use the iteratively regularized Gauss-Newton method (IRGNM) in BART to  directly estimate the MR parameter maps from undersampled k-space datasets. No pixel-wise fitting or intermediate reconstruction of contrast-weighted images is required!

For further information have a look into:

> Wang X, Roeloffs V, Klosowski J, Tan Z, Voit D, Uecker M, Frahm J.,  
[Model-based T1 Mapping with Sparsity Constraints Using Single-Shot Inversion-Recovery Radial FLASH](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26726).  
Magn Reson Med 2018;79:730-740.
<!-- #endregion -->

```bash id="OvICvnDSDwXn" colab={"base_uri": "https://localhost:8080/"} outputId="04d5cfb4-d9f5-438d-cb4f-a4cbaa6e3393"
bart moba -h
```

<!-- #region id="ejsKnyBDDwXo" -->
#### Coil Compression

Compress data to 3 coils, for good quality use 8 or more.

<!-- #endregion -->

```bash id="PeTLm8ZADwXo" colab={"base_uri": "https://localhost:8080/"} outputId="acdda4f9-3fb3-42ce-95ab-5d41b5a940a4"

NUM_VCOILS=3

## Coil compression
bart transpose 0 1 data/IR-FLASH ksp

# coil compression
bart cc -A -p $NUM_VCOILS ksp ksp_cc
```

<!-- #region id="2rqNkt2hDwXp" -->
#### Trajectory Preparation with Gradient Delay Correction
<!-- #endregion -->

```bash id="nQ8GLGjuDwXp" colab={"base_uri": "https://localhost:8080/"} outputId="c5a5bf80-6e96-4649-9a27-4392c99d6152"

READ=`bart show -d 0 data/IR-FLASH`
SPOKES=`bart show -d 1 data/IR-FLASH`
REP=`bart show -d 10 data/IR-FLASH`

## Prepare radial trajectory (golden-angle)
bart traj -r -c -D -G -x$READ -y$SPOKES -s7 -t$REP traj2

# Gradient Delay Correction
# Extract the steady-state data (data with less contrast change)
bart extract 10 920 1020 traj traj_extract
bart extract 10 920 1020 ksp_cc ksp_extract

# Transpose the 2nd and 10th dimension for the `estdelay` tool
bart transpose 10 2 traj_extract traj_extract1
bart transpose 10 2 ksp_extract ksp_extract1

# Gradient delay estimation usign RING
GDELAY=$(bart estdelay -R traj_extract1 ksp_extract1)

echo "Gradient Delays: "$GDELAY

# Calculate the "correct" trajectory with known gradient delays
bart traj -r -c -D -G -x$READ -y$SPOKES -s7 -t$REP -q $GDELAY trajn2

# 2x oversampling
bart scale -- 0.5 trajn2 trajn
```

<!-- #region id="S_N-Vq7sDwXq" -->
#### Preparation of Inversion Times
<!-- #endregion -->

```bash id="vuS4vYCJDwXq" colab={"base_uri": "https://localhost:8080/"} outputId="c605d9db-1683-4d49-d4eb-151857dc6909"

## Prepare time vector
TR=4100 #TR in [us]
BIN_SPOKES=20 # Bin data to save computation time 

# Read file dimensions from downloaded dataset
REP=`bart show -d 10 data/IR-FLASH`

NTIME=$((REP / BIN_SPOKES)) # Integer division!

# Create vector from 0 to NTIME
bart index 5 $NTIME tmp1
bart scale $(($BIN_SPOKES * $TR)) tmp1 tmp2
bart ones 6 1 1 1 1 1 $NTIME tmp1 
bart saxpy $((($BIN_SPOKES / 2) * $TR)) tmp1 tmp2 tmp3
bart scale 0.000001 tmp3 TI

# Reshape trajectory and data for model-based reconstruction
bart reshape $(bart bitmask 2 5 10) $BIN_SPOKES $NTIME 1 trajn traj_moba
bart reshape $(bart bitmask 2 5 10) $BIN_SPOKES $NTIME 1 ksp_cc ksp_cc_moba

# Resize data and trajectory for faster computation
bart resize -c 1 384 traj_moba traj_moba1
bart resize -c 1 384 ksp_cc_moba ksp_cc_moba1

echo "Trajectory:"
head -n2 traj_moba1.hdr

echo "Data:"
head -n2 ksp_cc_moba1.hdr

echo "TI:"
head -n2 TI.hdr''
```

<!-- #region id="dP5KouD8DwXs" -->
#### Nonlinear Model-based Reconstruction

The full nonlinear reconstruction can be applied to data by using only the `moba` command in the BART CLI. No coil sensitivity information is necessary, because they are jointly estimated. We apply a non-linear inversion-recovery Look-Locker model `-L` to our single-shot data. We also exploit compressed sensing by adding a wavelet $l_1$ regularization with the `-l1` flag.
<!-- #endregion -->

```bash id="L2qNDbGaDwXs" colab={"base_uri": "https://localhost:8080/"} outputId="9eb4efea-900f-4365-eb6d-92f3b3595152"

if $COLAB; then
    GPU=-g;
else
    GPU=''
fi

bart moba -L $GPU -d4 -l1 -i8 -C100 -j0.09 -B0.0 -n -t traj_moba1 ksp_cc_moba1 TI reco_moba 

#-L  --- to select look-locker model
#-g  --- to use GPU
#-d  --- debug level
#-l1 --- to use l1-Wavelet regularization
#-i  --- number of Newton-steps
#-C  --- maximum number of inner iterations (FISTA)
#-j  --- minimum regularization parameter
#-B  --- lower bound for relaxivity (R1s > 0)

# NOTE: There is no need of input of coil sensitivity maps, because we jointly estimate coils using model-based reconstruction
```

<!-- #region id="i-KXXZYZDwXs" -->
#### Visualize Results

To visualize the output of the reconstruction we resize it and thus remove the applied oversampling. Additionally, we slice the individual maps out of its original file and place them next to each other for the final visualization.
<!-- #endregion -->

```bash id="-XqJxYsEDwXt"

READ=`bart show -d 0 reco_moba`

# Remove oversampling for all maps
bart resize -c 0 $((READ / 2)) 1 $((READ / 2)) reco_moba reco_maps

# Merge all coefficients in the column dimension for visualization
bart reshape $(bart bitmask 1 6) $((3 * READ / 2)) 1 reco_maps reco_maps_lin

# Flip the maps in row dimension
bart flip $(bart bitmask 0) reco_maps_lin reco_maps_flip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 322} id="21rxunelmZTO" outputId="e4ef92c6-a424-41df-f5f8-152d45dba7a7"
! echo "Reconstructed Coefficient Maps: Mss, M0, R1s"
plot_map("reco_maps_flip", "viridis", 0, 3, '')
```

<!-- #region id="ygM6OQGHDwXt" -->
The output of the nonlinear Look-Locker model-based reconstruction are the parameter maps Mss, M0 and R1*.   To estimate the desired T1 map we pass the reconstruction to the `looklocker` command and visualize the T1 map.
<!-- #endregion -->

```bash id="va_-giUVDwXu"

INVERSION_DELAY=0.0153
THRESHOLD=0.2

# estimating T1 from the parameters  Mss, M0, R1s
bart looklocker -t $THRESHOLD -D $INVERSION_DELAY reco_maps tmp

# Flip the map in row dimension
bart flip $(bart bitmask 0) tmp moba_T1map
```

```python colab={"base_uri": "https://localhost:8080/", "height": 835} id="gDDc8HSOD8FQ" outputId="503fb366-3273-43d1-db73-b9cf592e0d43"
# python3 save_maps.py moba_T1map viridis 0 2.0 moba_T1map.png
plot_map("moba_T1map", "viridis", 0, 2, '$T_1$ / s')
```

<!-- #region id="xA8ldOwuDwXv" -->
### BART for Machine Learning - Reconstruction Networks

A specialized tutorial for neural networks in BART can be found in the [Workshop Material of the ISMRM 2021](https://github.com/mrirecon/bart-workshop/tree/master/ismrm2021).
<!-- #endregion -->

<!-- #region id="9GUPKERGrFnr" -->
#### Theory
We have implemented

> Variational Network<sup>1</sup>:
$$
x^{(i)} = x^{(i-1)}  - \lambda \nabla||Ax -b||^2 + Net(x^{(i-1)}, \Theta^{(i)} )
$$
> MoDL<sup>2</sup>:
$$
\begin{align}
z^{(i)} &= Net\left(x^{(i-1)}, \Theta \right)\\
x^{(i)} &= \mathrm{argmin}_x ||Ax -b||^2 + \lambda ||x - z^{(i)}||^2
\end{align}
$$

>Where
+ $A$ - MRI forward operator $\mathcal{PFC}$
    + $\mathcal{P}$ - Sampling pattern
    + $\mathcal{F}$ - Fourier transform
    + $\mathcal{C}$ - Coil sensitivity maps
+ $b$ - measured k-space data
+ $x^{(i)}$ - reconstruction after $i$ iterations
+ $x^{(0)}=A^Hb$ - initialization
+ $\Theta$ - Weights

>1: Hammernik, K. et al. (2018), [Learning a variational network for reconstruction of accelerated MRI data](https://doi.org/10.1002/mrm.26977). Magn. Reson. Med., 79: 3055-3071.

>2: Aggarwal, H. K. et al. (2019), [MoDL: Model-Based Deep Learning Architecture for Inverse Problems](https://doi.org/10.1109/TMI.2018.2865356). IEEE Trans. Med. Imag., 38(2): 394-405

To **train**, **evaluate** or **apply** unrolled networks, we provide the `bart reconet` command.
<!-- #endregion -->

```python id="gtzNi5vpDwXv" colab={"base_uri": "https://localhost:8080/"} outputId="18d92e2f-d429-41ae-c2b7-2dedc82340fd"
! bart reconet -h
```

```python id="h2T5krOSDwXw" colab={"base_uri": "https://localhost:8080/"} outputId="f7f27bef-d08f-4266-a0bc-99dd06c33d44"
! bart reconet --network h
```

<!-- #region id="dXvxFq6lDwXw" -->
#### Preparation of Knee-Data

Here, we use the data provided with the publication of the Variational Network, i.e. the coronal_pd_fs folder of the NYU-Dataset. The data has been converted to the .cfl-file format.   In the data folder, we find the fully-sampled kspace data of a knee and a sampling pattern. As the kspace is fully sampled, we can define a ground truth reference.

Before we apply the networks, we will create/estimate:
+ the downsampled kspace
+ coil sensitivity maps
+ a ground truth reference
<!-- #endregion -->

```python id="7HvKWTioDwXw" colab={"base_uri": "https://localhost:8080/", "height": 405} outputId="60f7581f-aa6c-4997-c9e8-15b1eb95a85d"
! echo $'\n# K-Space (fully sampled):'
! head -n2 data/kspace_fs.hdr

! echo $'\n# Pattern:'
! head -n2 data/pattern_po_4.hdr

pattern = np.abs(cfl.readcfl("data/pattern_po_4"))
plt.imshow(pattern, cmap="gray")
plt.show()
```

<!-- #region id="__DLJ4XLDwXx" -->
#### Create Downsampled Kspace

We downsample the fully-sampled kspace by multiplying it with the sampling pattern:
<!-- #endregion -->

```bash id="5-6QuQMsDwXx"
bart fmac data/kspace_fs data/pattern_po_4 kspace
```

<!-- #region id="rcv3PD0lDwXy" -->
#### Estimate Coil Sensitivity Maps

We estimate the coil sensitivity maps using ESPIRiT. 
<!-- #endregion -->

```bash id="dWOcX0FUDwXy" colab={"base_uri": "https://localhost:8080/"} outputId="8fc99d00-748a-4c91-c7d1-27f8e874657e"
bart ecalib -r24 -m1 kspace coils_l
bart resize -c 0 320 coils_l coils
```

<!-- #region id="dbWg5HJWDwXy" -->
#### Reconstruction of the Reference

We construct the **ground truth reference** as the coil-combinded reconstruction of the fully-sampled kspace data. For comparison, we also compute a **l1-wavelet** regularized and the **zero-filled** reconstruction.
<!-- #endregion -->

```bash id="g5l3sB0UDwXz" colab={"base_uri": "https://localhost:8080/"} outputId="473aef3e-7d13-4fcf-b387-734207e2c2f0"

mkdir -p tmp

FFT_FLAG=$(bart bitmask 0 1)
COIL_FLAG=$(bart bitmask 3)

# Reference
bart fft -i -u $FFT_FLAG data/kspace_fs tmp/coil_image
bart fmac -C -s$COIL_FLAG tmp/coil_image coils_l tmp/image

# PICS l1
bart pics -S -l1 -r0.003 -p data/pattern_po_4 kspace coils_l tmp/pics_reco_l

# Zero-filled
bart fft -i -u $FFT_FLAG kspace tmp/coil_image_zf
bart fmac -C -s$COIL_FLAG tmp/coil_image_zf coils_l tmp/image_zf_l

#resize (frequency oversampling)
bart resize -c 0 320 tmp/image ref
bart resize -c 0 320 tmp/pics_reco_l pics_reco
bart resize -c 0 320 tmp/image_zf_l zero_filled

rm -r tmp
```

<!-- #region id="0ciAHjCrDwX0" -->
We show the results:
<!-- #endregion -->

```python id="Pt06m9cUDwX0" colab={"base_uri": "https://localhost:8080/", "height": 464} outputId="66c797f8-e348-4c77-ebe6-63c5ad80a1c6"
ref = cfl.readcfl("ref")
pics_reco = cfl.readcfl("pics_reco")
zero_filled = cfl.readcfl("zero_filled")

vmax=0.5*np.max(np.abs(ref))

fig, axes = plt.subplots(figsize=(20,6), nrows=1, ncols=3, sharex=True, sharey=True)

axes[0].imshow(np.abs(ref[::-1,::-1]), cmap="gray", vmax=vmax)
axes[0].set_title("Coil Combined Reference", fontsize=20)

axes[1].imshow(np.abs(pics_reco[::-1,::-1]), cmap="gray", vmax=vmax)
axes[1].set_title("l1-Wavelet Regularized", fontsize=20)

axes[2].imshow(np.abs(zero_filled[::-1,::-1]), cmap="gray", vmax=vmax)
axes[2].set_title("Zero-filled Reconstruction", fontsize=20)

plt.tight_layout()
plt.show()
```

<!-- #region id="7Dv1UsVCDwX0" -->
#### Apply Variational Network

Having prepared the dataset, we can apply the Variational Network using the downloaded weights. The dataset is normalized by the maximum magnitude of the zero-filled reconstruction by using the `--normalize` option.  We use the pretrained weights provided in the weights directory. They have been trained on the first 15 knees from the coronal_pd_fs directory of the NYU-Dataset
<!-- #endregion -->

```bash id="xVg4RtKoDwX1" colab={"base_uri": "https://localhost:8080/"} outputId="88dbcdee-083e-4ca8-8808-19d2bd28ed24"

# if BART is compiled with gpu support, we add the --gpu option
if $COLAB; then
    GPU=--gpu;
else
    GPU=''
fi

bart reconet \
    $GPU \
    --network=varnet \
    --normalize \
    --apply \
    --pattern=data/pattern_po_4 \
    kspace \
    coils \
    data/varnet \
    varnet
```

<!-- #region id="SsP12TvLDwX1" -->
We plot the results:
<!-- #endregion -->

```python id="qXfmNgJADwX1" colab={"base_uri": "https://localhost:8080/", "height": 464} outputId="cd590cc1-51fb-48cb-caa4-f8de28074745"
ref = cfl.readcfl("ref")
pics_reco = cfl.readcfl("pics_reco")
varnet = cfl.readcfl("varnet")

vmax=0.5*np.max(np.abs(ref))

fig, axes = plt.subplots(figsize=(20,6), nrows=1, ncols=3, sharex=True, sharey=True)

axes[0].imshow(np.abs(ref[::-1,::-1]), cmap="gray", vmax=vmax)
axes[0].set_title("Coil Combined Reference", fontsize=20)

axes[1].imshow(np.abs(pics_reco[::-1,::-1]), cmap="gray", vmax=vmax)
axes[1].set_title("l1-Wavelet Regularized", fontsize=20)

axes[2].imshow(np.abs(varnet[::-1,::-1]), cmap="gray", vmax=vmax)
axes[2].set_title("Variational Network", fontsize=20)

plt.tight_layout()
plt.show()
```

<!-- #region id="zXkSd322DwX2" -->
#### Apply MoDL

Similarly, MoDL can be applied using the provided weights. Here, we unroll 5 iterations.
<!-- #endregion -->

```bash id="ogSm6QC8DwX2" colab={"base_uri": "https://localhost:8080/"} outputId="008843db-d0eb-4f2a-c75f-4e7ed9c09d6a"

# if BART is compiled with gpu support, we add the --gpu option
if $COLAB; then
    GPU=--gpu;
else
    GPU=''
fi

bart reconet \
    $GPU \
    --network=modl \
    --iterations=5 \
    --normalize \
    --apply \
    --pattern=data/pattern_po_4 \
    kspace \
    coils \
    data/modl \
    modl
```

<!-- #region id="RipDR2vQDwX2" -->
We plot the results:
<!-- #endregion -->

```python id="8Yyfi3DEDwX2" colab={"base_uri": "https://localhost:8080/", "height": 464} outputId="11730f8c-2a9f-4f66-a28a-e393eb55153f"
ref = cfl.readcfl("ref")
pics_reco = cfl.readcfl("pics_reco")
modl = cfl.readcfl("modl")

vmax=0.5*np.max(np.abs(ref))

fig, axes = plt.subplots(figsize=(20,6), nrows=1, ncols=3, sharex=True, sharey=True)

axes[0].imshow(np.abs(ref[::-1,::-1]), cmap="gray", vmax=vmax)
axes[0].set_title("Coil Combined Reference", fontsize=20)

axes[1].imshow(np.abs(pics_reco[::-1,::-1]), cmap="gray", vmax=vmax)
axes[1].set_title("l1-Wavelet Regularized", fontsize=20)

axes[2].imshow(np.abs(modl[::-1,::-1]), cmap="gray", vmax=vmax)
axes[2].set_title("MoDL", fontsize=20)

plt.tight_layout()
plt.show()
```

<!-- #region id="ncBxyjH9DwX2" -->
#### Evaluation of the Variational Network and MoDL
<!-- #endregion -->

```bash id="81wG5LfgDwX3" colab={"base_uri": "https://localhost:8080/"} outputId="d5087824-081a-46ea-f943-8c8cf5a9ef53"

# if BART is compiled with gpu support, we add the --gpu option
if $COLAB; then
    GPU=--gpu;
else
    GPU=''
fi

bart reconet \
    $GPU \
    --network=varnet \
    --normalize \
    --eval \
    --pattern=data/pattern_po_4 \
    kspace \
    coils \
    data/varnet \
    ref 
```

```bash id="4NayUikFDwX3" colab={"base_uri": "https://localhost:8080/"} outputId="4b24f71f-0c0e-4033-86bc-86cd267d9544"

# if BART is compiled with gpu support, we add the --gpu option
if $COLAB; then
    GPU=--gpu;
else
    GPU=''
fi

bart reconet \
    $GPU \
    --network=modl \
    --iterations=5 \
    --normalize \
    --eval \
    --pattern=data/pattern_po_4 \
    kspace \
    coils \
    data/modl \
    ref 
```

```python id="2KiJMXBZKvIw"

```
