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

---

Create the input data. We will use the Shepp Logan `phantom` tool to create 
multi-channel k-space data, add noise with the `noise` tool, and sample the data using
the Poisson disc `poisson` tool.

First set the parameters. The `-x` flag specifies the 2D dimensions. The `-k` flag creates the phantom directly in k-space.
The `-s` flag specifies the number of coil sensitivities.
```bash
nmaps=8 # coil sensitivities
ncc=6 # coil compression
dim=256 # ky/kz dims
bart phantom -x $dim -k -s $nmaps ksp_orig
```

To match to the MRI-specific tools, we tranpose the spatial dimensions
```
bart reshape 7 1 $dim $dim ksp_orig ksp_orig
```

Add noise
```bash
noisevar=100
bart noise -n $noisevar ksp_orig ksp_noise
```

FFT and root-sum-of-square combine the noisy data
```bash
bart fft -iu 7 ksp_noise cimg_noise
bart rss 8 cimg_noise img_noise
rmcfl cimg_noise
```

Create a variable density Poisson disc sampling pattern and sample k-space. The acceleration factor is given by `yaccel` and `zaccel`.
A fully sampled calibration region of size `caldim` is added.
```bash
yaccel=1.5
zaccel=1.5
caldim=32
bart poisson -Y $dim -Z $dim -y $yaccel -z $zaccel -C $caldim -v -e mask
bart fmac ksp_noise mask ksp_und
```

View the mask
```bash
cflview mask &
```

![](images/mask.png?raw=true)


---

Coil-compress and reconstruct the data. We will generate ESPIRiT sensitivity maps using the `ecalib` tool. We will use
the `pics` to compare various ESPIRiT-based parallel imaging and compressed sensing reconstructions. This includes the
following:
* ESPIRiT with Tikhonov regularization
* L1-ESPIRiT with l1 wavelet regularization
* L1-ESPIRiT with total tariation
* Zero-filled reconstruction

Coil compression
```bash
bart cc -p $ncc ksp_und ksp_und_cc
```

ESPIRiT sensitivity map calibration
```bash
bart ecalib -S ksp_und_cc maps
```

View the sensitivities
```bash
cflview maps &
```
![](images/sens.png?raw=true)

ESPIRiT reconstruction
```bash
l2=.01
bart pics -d5 -R Q:$l2 -S ksp_und_cc maps img_recon0
bart slice 4 0 img_recon0 recon_l2 # remove the extra map
```

L1-ESPIRiT with Wavelets
```bash
l1wav=.008
bart pics -d5 -S -R W:7:0:$l1wav ksp_und_cc maps img_recon0
bart slice 4 0 img_recon0 recon_wav # remove the extra map
```

L1-ESPIRiT with Total Variation
```bash
l1tv=.008
bart pics -d5 -S -R T:7:0:$l1tv ksp_und_cc maps img_recon0
bart slice 4 0 img_recon0 recon_tv # remove the extra map
rmcfl img_recon0
```

Zero-fill reconstruction
```bash
bart fft -iu 7 ksp_und_cc cimg
bart rss 8 cimg img_zf
rmcfl cimg
```

Compare the results
```bash
bart join 2 img_noise img_zf recon_l2 recon_wav recon_tv img_compare
cflview img_compare &
```
![](images/img_compare.png?raw=true)
