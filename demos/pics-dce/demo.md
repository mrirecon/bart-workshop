First let's add BART to our system path. Make sure `TOOLBOX_PATH` is set to the location of BART.
```bash
export PATH=${TOOLBOX_PATH}:${PATH}
```

Define our viewer to save images to png
```bash
VIEWER="bart toimg"
function cflview () {
	if [[ $VIEWER == "bart toimg" ]] ; then
		$VIEWER $1 $1.png
	else
		$VIEWER $1
	fi
}
```

Add a convenience function for deleting BART files
```bash
function rmcfl () {
	rm -f $1.cfl
	rm -f $1.hdr
}
```


Add a convenience function to perform FFT-interpolation
```bash
function fft_interp () {
	bart fft -u 7 $2 $2.tmp
	bart resize -c 1 $1 $2.tmp $3.tmp
	bart fft -iu 7 $3.tmp $3
	
	rmcfl $2.tmp
	rmcfl $3.tmp
}


In this demo we will investigate prospectively under-sampled dynamic contrast enhanced (DCE) MRI data provided by [Tao
Zhang](http://web.stanford.edu/~tzhang08/). Unzip the data and it will be placed under `data/`.
```bash
unzip data.zip
```

The data consist of a single slice from under-sampled multi-channel, Cartesian k-space binned into temporal phases, as well as
ESPIRiT sensitivity maps and sampling weights derived
from navigator data. We can see the dimensions using the `show` command:
```bash
bart show -m data/ksp
#Type: complex float
#Dimensions: 16
#AoD:	1	68	180	20	1	1	1	1	1	1	18	1	1	1	1	1
```
(we can also use the simpler command, `cat data/ksp.hdr`, as the header file is plain-text and describes the
dimensions. However, the header file may change in the future so it is safer to use the `show` command.)

The MRI-specific BART commands assume that the first three dimensions represent space/k-space, the next two dimensions
represent coils and ESPIRiT maps, and the 10th dimensions (zero-indexing) represents temporal phases.
We see the spatial dimensions are `1 X 68 x 180`, with 20 coils and 18 temporal phases.

---

To start, let's compute the sampling pattern and compare it to the k-space weighting mask.
The weighting mask assigns each phase encode a weight between zero and one, depending on the amount of
motion-corruption.
```bash
bart pattern data/ksp data/pat # generate a sampling pattern from k-space
bart join 2 data/pat data/weights data/pat_weights
bart show -m data/pat_weights
#Type: complex float
#Dimensions: 16
#AoD:	1	68	360	1	1	1	1	1	1	1	18	1	1	1	1	1
```

We can sum along the temporal dimension using the `fmac` (fused multiply accumulate) tool:
```bash
bart fmac -h
#Usage: fmac [-C] [-s d] <input1> <input2> <output>
#
#Multiply and accumulate.
#
#-C		conjugate input2
#-s b      	squash dimensions selected by bitmask b
#-h		help
```
The bitmask is used to specify the dimensions we want to sum over. We can quickly compute a bitmask using the `bitmask`
tool:
```bash
bart bitmask 10 # bitmask for the temporal dimensions
#1024
bart bitmask 0 1 2 # bitmask for 3D spatial dimensions
#7
```

To sum over the temporal dimension, we create a BART file containing a single entry with value one, and sum over the
10th dimension:
```bash
bart ones 1 1 o
bart fmac -s $(bart bitmask 10) data/pat_weights data/pat_weights_sum
rmcfl o
cflview data/pat_weights_sum & # visualize the result
```

---


We will use the `pics` tool to perform ESPIRiT-based parallel imaging and compressed sensing. The basic usage is
```bash
bart pics [optimization options] [regularization options] kspace maps recon
```
See the full list with `bart pics -h`.
There are several built-in regularization terms and transforms. We can see the options by invoking the help:
```bash
bart pics -Rh
#Generalized regularization options (experimental)
#
#-R <T>:A:B:C	<T> is regularization type (single letter),
#		A is transform flags, B is joint threshold flags,
#		and C is regularization value. Specify any number
#		of regularization terms.
#
#-R Q:C    	l2-norm in image domain
#-R I:B:C  	l1-norm in image domain
#-R W:A:B:C	l1-wavelet
#-R T:A:B:C	total variation
#-R T:7:0:.01	3D isotropic total variation with 0.01 regularization.
#-R L:7:7:.02	Locally low rank with spatial decimation and 0.02 regularization.
#-R M:7:7:.03	Multi-scale low rank with spatial decimation and 0.03 regularization.
```

First let's reconstruct with l1-wavelets in space and total variation in time.
```bash
bart pics -i 100 -p data/weights -R T:$(bart bitmask 10):0:.02 -R W:$(bart bitmask 0 1 2):0:0.001 data/ksp data/maps recon_wavtv
bart slice 4 0 recon_wavtv tmp001 # throw out second ESPIRiT image
bart flip 0 tmp001 recon_wavtv
rmcfl tmp001
```

The `y` direction of the image is quite low. We can perform FFT interpolation along the first dimension.  
```bash
fft_interp 80 recon_wavtv recon_wavtv
```

Next let's reconstruct using locally low rank (LLR).
```bash
bart pics -i 100 -p data/weights -R L:$(bart bitmask 0 1 2):0:0.05 data/ksp data/maps recon_llr
bart slice 4 0 recon_llr tmp001 # throw out second ESPIRiT image
bart flip 0 tmp001 recon_llr
rmcfl tmp001
fft_interp 80 recon_llr recon_llr
```

Combine the reconstructions and compare
```bash
bart join 1 recon_wavtv recon_llr recon_compare # combine the reconstructions
cflview recon_compare &
```
