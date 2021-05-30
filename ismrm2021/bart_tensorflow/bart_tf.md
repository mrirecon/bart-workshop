---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="11904043" -->
# Enhance BART with a TF Computation Graph

**Authors**: [Guanxiong Luo](mailto:guanxiong.luo@med.uni-goettingen.de), [Nick Scholand](mailto:nick.scholand@med.uni-goettingen.de), [Christian Holme](mailto:christian.holme@med.uni-goettingen.de)

**Presenter**: [Guanxiong Luo](mailto:guanxiong.luo@med.uni-goettingen.de)

**Institution**: University Medical Center Göttingen

**Reference**:
> Luo, G, Blumenthal, M, Uecker, M. Using data-driven image priors for image reconstruction with BART Proc. Intl. Soc. Mag. Reson. Med. 29 (2021) P.1756

## Overview
This tutorial is to present how to create a regularization term with tensorflow and use it for image reconstruction in [BART](https://github.com/mrirecon/bart).

<img src="https://github.com/mrirecon/bart-workshop/raw/master/ismrm2021/bart_tensorflow/over.png" width="800"/>

## What we have
TensorFlow provides a C API that can be used to build bindings for other languages. 

1. BART src/nn/tf_wrapper.c

    * create tensors, create tf session

    * import the exported graph

    * restore the session from the saved model

    * get operation nodes from the graph

    * execute operation with session.run()


2. TensorFlow C Libraries [2.4.0](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz)

3. A python program to export graph and weights (if any)

## What you can do with tf graph

> We can create the regularization term $R(x)$ with tf graph for image reconstruction (integrated in BART's `pics` tool).

$$\underset{x}{\arg \min}\ \|Ax-y\|^2+\lambda R(x)$$

## What you can learn here

1. simple example $R(x)=\|x\|^2$ without trainable weights

2. $R(x)=\log p(x, net(\Theta,x))$ with trainable weights $\Theta$, $net$ is represented as a prior [1]

[1] Luo, G, Zhao, N, Jiang, W, Hui, ES, Cao, P. MRI reconstruction using deep Bayesian estimation. Magn Reson Med. 2020; 84: 2246– 2261. https://doi.org/10.1002/mrm.28274
<!-- #endregion -->

<!-- #region id="0xulGSZsypXz" -->
## Part 0: Download Supporting Material
<!-- #endregion -->

<!-- #region id="h2blGxLdypXz" -->
This tutorial requires additional data including radial k-space spokes, a trained model and some python functions. If you want to follow up this tutorial execute the following cell, which downloads the required files.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bU44WbkpypX0" outputId="2351faee-13f5-4368-f731-1c7b0aef53c9"
# Download the required supporting materials
! wget -q https://raw.githubusercontent.com/mrirecon/bart-workshop/master/ismrm2021/bart_tensorflow/data.zip
! unzip data.zip
```

<!-- #region id="3b544b17" -->
## Part I: How to Create a TF Graph for BART

The first part of this tutorial is about creating a TF graph, which can be used with BART. Therefore, we need load some python libraries.
<!-- #endregion -->

```python id="d9a21c71"
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np
```

<!-- #region id="30b64c57" -->
### Step 1: Define Input $x$

We define a TF object for our input.
<!-- #endregion -->

```python id="d4d77905"
image_shape = [256, 256, 2]
batch_size = 1

# CAPI -> TF_GraphOperationByName(graph, "input_0")
# give name with input_0, ..., input_I 
x = tf.placeholder(tf.float32,
                   shape=[batch_size]+image_shape,
                   name='input_0')
v = tf.Variable(1.)
x = x * v
```

<!-- #region id="8e2719ae" -->
### Step 2: Set Output to TF's l2 Loss

In the later following reconstruction example we want to validate the integration of TF graphs in BART with an l2 regularization example. 

Therefore, we are going to compare a reconstruction with the internal l2 loss of the `pics` tool with the result using a TF graph regularization.

The required graph is now set to have TF's l2 loss as an output.
<!-- #endregion -->

```python id="f2920129"
l2 = tf.nn.l2_loss(x)#/np.product(image_shape)/batch_size        #R(x)=|x|^2
# CAPI -> TF_GraphOperationByName(graph, "output_0") -> nlop forward
# give name with output_0, ..., output_I
output = tf.identity(tf.stack([l2, tf.ones_like(l2)], axis=-1), name='output_0') 
```

<!-- #region id="6e8f00fa" -->
### Step 3: Define the Gradient of TF's l2 Loss

For being able to use the TF graph as a regularization inside of a larger iterative optimization algorithm, we need to know its gradients, which are defined in the following.
<!-- #endregion -->

```python id="2d24fc05"
grad_ys = tf.placeholder(tf.float32,
                         shape=[2],
                         name='grad_ys_0')

# CAPI -> TF_GraphOperationByName(graph, "grad_0") -> nlop adj
grads = tf.squeeze(tf.gradients(output, x, grad_ys), name='grad_0') 
```

<!-- #region id="88c848b3" -->
### Step 4: Export the Graph and Weights

The created l2 loss graph needs to be stored in a BART understandable format, which allows us to pass it to the `pics` command of BART's CLI. Therefore, we exploit the `export_model` function.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="b0acc0c0" outputId="c132aef9-07ac-4342-b647-0504b37fbd34"
from utils import export_model

# Definition of function:
#         export_model(model_path, exported_path, name, as_text, use_gpu):

export_model(None, "./", "l2_toy", as_text=False, use_gpu=False)
```

<!-- #region id="bbzuEChGypX4" -->
Let us have a look how the exported files look like.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="49f7dc3f" outputId="60e57c23-6a2d-42b3-e167-e785e8b03750"
!ls
```

<!-- #region id="ce4a7796" -->
##  Part II: How to Use the TF Graph in BART
<!-- #endregion -->

<!-- #region id="f1bfbc28" -->
###  Step 1: Setup BART and TF
<!-- #endregion -->

<!-- #region id="Dokk3t5zypX5" -->
#### TF C API

First we need to **download the TF C API**
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="1ea147d2" outputId="1cdd54c5-9017-4479-b3a3-183e66d48340"

# Download tensorflow c libraries
wget -q https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz
mkdir tensorflow && tar -C tensorflow -xvzf libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz
```

<!-- #region id="RyKtOniHypX5" -->
and need to set the required environmental variables
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1cOk227HH3Vv" outputId="e9973ea8-f0d4-4d07-9380-ed83eaa48630"
%env LIBRARY_PATH=/content/tensorflow/include 
%env LD_LIBRARY_PATH=/content/tensorflow/lib
%env TF_CPP_MIN_LOG_LEVEL=3
```

<!-- #region id="MYceQ2_XypX6" -->
#### Download and Compile BART
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="6NaibCzaypX6" outputId="4176f00f-18b9-4c54-b459-698f1bcb485b"

# Install BARTs dependencies
apt-get install -y make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev &> /dev/null

# Download BART version

[ -d /content/bart ] && rm -r /content/bart
git clone https://github.com/mrirecon/bart/ bart

[ -d "bart" ] && echo "BART was downloaded successfully."
```

<!-- #region id="LNmLP7YSypX7" -->
After downloading BART we need to compile it. Make sure the flags

- `TENSORFLOW=1`
- `TENSORFLOW_BASE=../tensorflow/`,

which are required to intgrate TF graphs in BART, are set.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="ReAYicSoypX7" outputId="7f553ed1-dda4-4dde-a878-ec4fe5233db0"

cd bart

# Switch to desired branch of the BART project
BRANCH=master
git checkout $BRANCH

# Define specifications 
COMPILE_SPECS=" PARALLEL=1
                TENSORFLOW=1
                TENSORFLOW_BASE=../tensorflow/"

printf "%s\n" $COMPILE_SPECS > Makefiles/Makefile.local

make &> /dev/null
```

<!-- #region id="j8H8BNqYypX7" -->
After compilation of BART we need to set the required environmental variable: `TOOLBOX_PATH`
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jwyGBFWDypX8" outputId="6ba2b2f6-6738-4170-c6f4-e224f4a8eca3"
%env TOOLBOX_PATH=/content/bart
```

<!-- #region id="mjJct8zTypX8" -->
Additionally, we add the compiled `bart` executable to our `PATH` variable
<!-- #endregion -->

```python id="0RI9l6blElDF"
import os
os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
```

<!-- #region id="743c3f4a" -->
### Step 2: Help Information for TF Graph in BART's `pics`

In the second step we look into the help for BART's regularization options for the `pics` tool.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1f613571" outputId="fe50220b-11c3-4593-a577-799f5ceb8d9f"
!bart pics -Rh
```

<!-- #region id="dd4b2e2a" -->
You can find a list of regularization terms $R(x)$, which can be added to the optimization of
$$
\hat{x}=\underset{x}{\arg \min} \|x-v\|^2 + \lambda R(x)
$$

To integrate a TF graph as regularization term in `pics` use the notation `-R TF:{graph_path}:lambda`.
<!-- #endregion -->

<!-- #region id="0e14c927" -->
### Step 3: Extract Radial Spokes and Compute Coil Sensitivities
<!-- #endregion -->

<!-- #region id="2SY7QAIjypX9" -->
The dataset we downloaded provides us with radial k-space data consisting of 160 spokes following a sampling scheme rotated by the 7th golden angle.

For this tutorial we will use the first 60 spokes and extract them from the original dataset.
<!-- #endregion -->

```bash id="7942ccc3"

# Extract spokes from original dataset

spokes=60

bart extract 2 0 $spokes ksp_256 ksp_256_c
bart extract 2 0 $spokes traj_256 traj_256_c
```

<!-- #region id="gD4GvsY6ypX-" -->
To be able to exploit ESPIRiT for coil sensitivity estimation, we need to grid the non-Cartesian (radial) dataset. Instead of gridding it directly we use the internal gridding of the inverse `nufft` tool and project the result back into k-space with a regular `fft`.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="7ca46fd4" outputId="ec026738-7f6a-4f30-cd25-026571995017"

# Grid non-Cartesian k-space data

bart nufft -i traj_256_c ksp_256_c zero_filled
bart fft $(bart bitmask 0 1) zero_filled grid_ksp

```

<!-- #region id="0Tkx4-j2ypX_" -->
After gridding the radial dataset, we can use ESPIRiT to estimate the coil sensitivity maps.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="Rpyx22-FypX_" outputId="973d7d17-05be-492e-88fc-c59c851ff0d3"

# Estimate coil-sensitivities with ESPIRiT

bart ecalib -r20 -m1 -c0.0001 grid_ksp coilsen_esp
```

<!-- #region id="3fcd3a99" -->
## Example 1: TF Graph as l2 Regularization
<!-- #endregion -->

<!-- #region id="US_EtZpEypX_" -->
In this first example we use the TF graph as an l2 regularization term. We pass the TF graph following the `-R TF:{graph_path}:lambda` notation for `pics` regularization terms.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="334b8029" outputId="ef514ce1-60c6-4886-f78d-0f8da05a12bf"
# Reconstruct with TF Graph as l2 regularization

!bart pics -i100 -R TF:{$(pwd)/l2_toy}:0.02 -d5 -e -t traj_256_c ksp_256_c coilsen_esp l2_pics_tf
```

<!-- #region id="qqpVGocNypYA" -->
After reconstruction of the dataset with the TF graph as loss we validate it by comparing it to the built-in l2 regularization in `pics` called by adding the `-l2` flag to the CLI call.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4d9f2471" outputId="11efb117-12ef-43fb-e98b-e47d07c310d0"
# Reconstruct with built-in l2 regularization

!bart pics -l2 0.01 -e -d5 -t traj_256_c ksp_256_c coilsen_esp l2_pics
```

<!-- #region id="2lAxSDPVypYA" -->
For an improved comparison we visualize them next to each other.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="029c7626" outputId="47bd6c54-df8a-48ab-f0e9-408b1d53f4f6"
from utils import *
import matplotlib.pyplot as plt
fig, axis = plt.subplots(figsize=(8,4), ncols=2)
l2_pics = readcfl("l2_pics")
l2_pics_tf = readcfl("l2_pics_tf")

axis[0].imshow(abs(l2_pics), cmap='gray', interpolation='None')
axis[1].imshow(abs(l2_pics_tf), cmap='gray', interpolation='None')
axis[0].set_title("l2_pics", fontsize=20)
axis[1].set_title("l2_pics_tf", fontsize=20)
axis[0].axis('off')
axis[1].axis('off')
```

<!-- #region id="da05b37c" -->
## Example 2: $R(x)=\log p(x, net(x))$ 
<!-- #endregion -->

<!-- #region id="MVfZh49mypYB" -->
After validating the TF graph import as regularization based on the l2 regularization, we want to add knowledge from a trained prior to our reconstruction.

We already downloaded the pretrained prior and can have a look into its files
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="a34231d2" outputId="c5bbf567-2433-4d41-bee9-c22ffa4a405d"
!ls prior/
```

<!-- #region id="hHiREScTypYC" -->
We need to generate weights for a density compensation in the following reconstruction. This can be done with the `gen_weights` function. To use the result from BART's CLI we need to convert it into the .cfl format. Here, we use the `writecfl` function of BART's python interface.
<!-- #endregion -->

```python id="edc18970"
writecfl("weights", gen_weights(60, 256))
```

<!-- #region id="s5HXpKCbypYD" -->
Now we can run the reconstruction.

Again we pass the information about our regularization using the `-R TF:{graph_path}:lambda` notation of `pics` regularization. We run the reconstruction for 30 iterations.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9b0b02b9" outputId="4308a74e-6554-4c3f-b303-a7c4ce255cb3"
!bart pics -i30 -R TF:{./prior/pixel_cnn}:8 -d5 -e -I -p weights -t traj_256_c ksp_256_c coilsen_esp w_pics_prior
```

<!-- #region id="--9WFDSVypYD" -->
Finally, we can compare the results for the built-in l2, TF l2 and prior regularized reconstruction with the `pics` tool in BART. We visualize all results next to each other.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 266} id="7c32249b" outputId="f07cb662-1c72-41da-d033-9a069a965f05"
import matplotlib.pyplot as plt
pics_prior = readcfl("w_pics_prior")
fig, axis = plt.subplots(figsize=(12,4), ncols=3)

axis[0].imshow(abs(l2_pics), cmap='gray', interpolation='None')
axis[0].set_title("l2_pics", fontsize=20)
axis[1].imshow(abs(l2_pics_tf), cmap='gray', interpolation='None')
axis[1].set_title("l2_pics_tf", fontsize=20)
axis[2].imshow(abs(pics_prior), cmap='gray', interpolation='None')
axis[2].set_title("prior_pics", fontsize=20)
axis[0].axis('off')
axis[1].axis('off')
axis[2].axis('off')

```

```python colab={"base_uri": "https://localhost:8080/"} id="2f804e22" outputId="85864854-f6e0-40ec-8a0f-4faad7d2222b"
! bash clean
```

```python id="J66627RqypYE"

```
