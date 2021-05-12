---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.8.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="0rvboizZrO00" -->
# BART on Google Colab

**Author**:  
Nick Scholand, [nick.scholand@med.uni-goettingen.de](mailto:nick.scholand@med.uni-goettingen.de)  
Christian Holme, [christian.holme@med.uni-goettingen.de](mailto:christian.holme@med.uni-goettingen.de)


**Institution**: University Medical Center Göttingen


## Choose GPU
  
Choose a GPU instance for this tutorial:

- Go to Edit → Notebook Settings
- choose GPU from Hardware Accelerator drop-down menu

<!-- #endregion -->

<!-- #region id="xdtZk9QoTPEK" -->
## Check GPU Hardware

In the beginning of this tutorial we need to check which GPU type we got from Google Colab. Free users are provided with either a **Tesla T4**, a **Tesla P100-PCIE-16GB** or a **Tesla K80** GPU.

While the Tesla T4 and Tesla P100-PCIE-16GB support the default CUDA 11 version, the Tesla K80 does not.

If you got a K80 assigned to, the following lines will reset your default CUDA to version 10.1, which should work afterwards.
<!-- #endregion -->

```bash id="w6GGdNwaAL-q" outputId="a989ff40-cd7e-47c2-d09d-d85f2f51e8ad" colab={"base_uri": "https://localhost:8080/"}

# Estimate GPU Type
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

echo "GPU Type:"
echo $GPU_NAME

if [ "Tesla K80" = "$GPU_NAME" ];
then
    echo "GPU type Tesla K80 does not support CUDA 11. Set CUDA to version 10.1."

    # Install CUDA-10.1 if not already installed
    apt-get install cuda-10-1 cuda-drivers &> /dev/null

    # Change default CUDA to version 10.1
    cd /usr/local
    rm cuda
    ln -s cuda-10.1 cuda

else
    echo "GPU Information:"
    nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv
    nvcc --version
    echo "Current GPU supports default CUDA-11."
    echo "No further actions are necessary."
fi
```

```python id="XK9dbH0xzDeK"
import os

# Set Library path for current CUDA version
os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda/lib64"
```

<!-- #region id="p7vVofjBTiOO" -->
## Install BART

If you have BART already installed on your system, [skip →](#python) BARTs installation part.

To install [BART](https://github.com/mrirecon/bart/tree/master) we need some special dependencies. Let us install them on your chosen system.
<!-- #endregion -->

```bash id="YgW3_QiATvXH"

# Install BARTs dependencies

apt-get install -y make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev &> /dev/null
```

<!-- #region id="zAAQiQP6rtjr" -->
After installing BARTs dependencies we now want to get the latest BART version from its [GitHub repository](https://github.com/mrirecon/bart/tree/master).
<!-- #endregion -->

```bash id="lcSmjqh1qyUs" outputId="77e82336-8940-4876-a5fc-128a32b7883b" colab={"base_uri": "https://localhost:8080/"}

# Download BART version

[ -d /content/bart ] && rm -r /content/bart
git clone https://github.com/mrirecon/bart/ bart

[ -d "bart" ] && echo "BART branch ${BRANCH} was downloaded successfully."
```

<!-- #region id="FLsc1DASsItD" -->
In the next step we are going to compile BART. Ensure that the correct references to your CUDA instance 

- `CUDA`
- `CUDA_BASE`
- `CUDA_LIB`

are chosen! The default is set to work on [Google Colab](https://colab.research.google.com).
<!-- #endregion -->

```bash id="CuGRjSoechKI" outputId="780fe0c7-924b-41b4-c590-a0300ef92af1" colab={"base_uri": "https://localhost:8080/"}

cd bart

# Switch to desired branch of the BART project
BRANCH="master"
git checkout $BRANCH

# Define specifications 
COMPILE_SPECS=" PARALLEL=4
                CUDA=1
                CUDA_BASE=/usr/local/cuda
                CUDA_LIB=lib64"

printf "%s\n" $COMPILE_SPECS > Makefiles/Makefile.local

# Compile BART with CUDA
make &> /dev/null

cd ..
```

<!-- #region id="X1rIsqIWX_kP" -->
## Set Environment for BART

After downloading and compiling BART, the next step simplifies the handling of BARTs command line interface inside of a ipyhton jupyter-notebook.


<!-- #endregion -->

```python id="lxohd5PMQOi4"
import os
import sys

# Define environment variables for BART and OpenMP

os.environ['TOOLBOX_PATH'] = "/content/bart"

os.environ['OMP_NUM_THREADS']="4"

# Add the BARTs toolbox to the PATH variable

os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
sys.path.append(os.environ['TOOLBOX_PATH'] + "/python")
```

<!-- #region id="il-AmS98akEz" -->
Now BART can be called by
<!-- #endregion -->

```bash id="6E7Kb0jPancG" outputId="532e7fdd-897f-468e-a769-63a589f032e1" colab={"base_uri": "https://localhost:8080/"}
bart
```

<!-- #region id="MauNJ4OwY1xQ" -->
<a name="python"></a>
## Install Python Dependencies


Additionally to the installation of BART, we need some python libraries for visualizations and data handling.
<!-- #endregion -->

```python id="YIWaPpImY6-3"
import cfl
import numpy as np

from IPython.display import Image
```

<!-- #region id="x4IWlOoY1D47" -->
# Example: Running BART on Google Colabs GPUs
<!-- #endregion -->

```bash id="moud5YN_F1ka" outputId="f710c76f-82d3-4245-dad6-230629898121" colab={"base_uri": "https://localhost:8080/"}

SAMPLES=384
SPOKES=91

# Create a radial trajectory to work with

bart traj -x $SAMPLES -y $SPOKES -r -D -l t
bart scale 0.5 t t2

head -n2 t2.hdr
```

```bash id="ygadTOJto_mn" outputId="f1b84e48-6c16-4c49-a88f-b96fb3673505" colab={"base_uri": "https://localhost:8080/"}

# Create a Cartesian phantom

bart phantom -k -s 8 -t t2 k

head -n2 k.hdr
```

```bash id="V0jDQ1mh33GR" outputId="9001727f-21c1-4537-bb70-130ff3f59f95" colab={"base_uri": "https://localhost:8080/"}

# Estimate the perfect coil-profiles

bart phantom -x192 -S8 coils

head -n2 coils.hdr
```

```bash id="Noir8EbV1T2c" outputId="6945050f-3cac-44c5-d405-09a4f12a8b3d" colab={"base_uri": "https://localhost:8080/"}

# Reshape Coils to plot 2D image
bart reshape $(bart bitmask 1 3) $((192*8)) 1 coils coils_flat

bart toimg -W coils_flat coils_flat.png
```

```python id="NTyg_Kjlb44q" outputId="45ff8dbe-966a-42fb-f77b-d78e867809c5" colab={"height": 209, "base_uri": "https://localhost:8080/"}
# Visualize coils
Image('coils_flat.png')
```

```bash id="RBfLfs_yobK4" outputId="8dbd7e15-069d-4557-91c4-7ad2fc95bac4" colab={"base_uri": "https://localhost:8080/"}

# Perform an nuFFT reconstruction on the GPU...

bart nufft -g -i t2 k reco_nufft
head -n2 reco_nufft.hdr

# ...and combine all coil images

bart rss $(bart bitmask 3) reco_nufft reco_n
head -n2 reco_n.hdr
```

```python id="gjQDIv9m2E77" outputId="5459565e-31de-4559-f219-3d8afe3edcde" colab={"height": 227, "base_uri": "https://localhost:8080/"}
# Visualize nufft reconstruction
!bart toimg -W reco_n reco_n.png
Image('reco_n.png')
```

```bash id="TX7xi-Doo0-q" outputId="e6903c84-7d69-41a9-ae93-ad78baa90b6b" colab={"base_uri": "https://localhost:8080/"}
# Perform a Parallel-Imaging Compressed Sensing Reconstruction on GPU
bart pics -g -e -t t2 k coils reco_pics
```

```python id="48QpSeSF2hdV" outputId="e54ff377-e8b5-43d1-d8bd-f0c57ebc30b2" colab={"height": 227, "base_uri": "https://localhost:8080/"}
# Visualize PICS reconstruction
!bart toimg -W reco_pics reco_pics.png
Image('reco_pics.png')
```

```bash id="9oQRmuKcGEu5"

# Clean up
rm *.cfl *.hdr
```

```python id="vHpUPtvH6hzx"

```
