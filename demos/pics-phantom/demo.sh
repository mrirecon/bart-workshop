#!/bin/bash

set -eu
set -o pipefail

set -x

function handle_signal() {
    exit 1
}
trap handle_signal SIGABRT SIGTERM SIGINT SIGHUP SIGSEGV

VIEWER="bart toimg"
VIEWER=bartview.py
#VIEWER=viewer
function cflview () {
	if [[ $VIEWER == "bart toimg" ]] ; then
		$VIEWER $1 $1.png
	else
		$VIEWER $1
	fi
}

function rmcfl () {
	rm -f $1.cfl
	rm -f $1.hdr
}


## ----------------------------------------------------------
# Create the input data. We will use the Shepp Logan phantom tool to create
# multi-channel k-space data, add noise, and sample it with the Poisson disc tool

# Generate a 2D Shepp Logan phantom in k-space
nsens=8 # coil sensitivities
ncc=6 # coil compression
dim=256 # ky/kz dims
bart phantom -x $dim -k -s $nsens ksp_orig

# Add noise 
noisevar=100
bart noise -n $noisevar ksp_orig ksp_noise

# FFT and root-sum-of-square combine the noisy data
bart fft -iu 7 ksp_noise cimg_noise
bart rss 8 cimg_noise img_noise
rmcfl cimg_noise


# Create a Poisson disc sampling pattern and sample k-space
yaccel=1.5
zaccel=1.5
caldim=32
bart poisson -Y $dim -Z $dim -y $yaccel -z $zaccel -C $caldim -v -e mask
bart fmac ksp_noise mask ksp_und

# View the mask
cflview mask &


## ----------------------------------------------------------
# Coil-compress the data and perform an L1-ESPIRiT reconstruction

# Coil compression
bart cc -P $ncc ksp_und ksp_und_cc

# ESPIRiT sensitivity map calibration
bart ecalib -S ksp_und_cc sens

# View the sensitivities
cflview sens &

# ESPIRiT reconstruction
l2=.01
bart pics -R Q:$l2 -S ksp_und_cc sens img_recon0
bart slice 4 0 img_recon0 recon_l2 # remove the extra map

# L1-ESPIRiT with Wavelets
l1wav=.008
bart pics -S -R W:7:0:$l1wav ksp_und_cc sens img_recon0
bart slice 4 0 img_recon0 recon_wav # remove the extra map

# L1-ESPIRiT with Total Variation
l1tv=.008
bart pics -S -R T:7:0:$l1tv ksp_und_cc sens img_recon0
bart slice 4 0 img_recon0 recon_tv # remove the extra map
rmcfl img_recon0

# Zero-fill reconstruction
bart fft -iu 7 ksp_und_cc cimg
bart rss 8 cimg img_zf
rmcfl cimg

# Compare the results
bart join 2 img_noise img_zf recon_l2 recon_wav recon_tv img_compare
cflview img_compare &


# Coil combine
fft -i -u 6 kSlice coil_images
fmac -C -s 8 coil_images map im

# TV denoise
rof 20000 6 im im2

# Coil split
fmac map im2 coil_images2
fft -u 6 coil_images2 kSlice2

# Data consistency

pattern kSlice pat

ones 3 1 256 184 uno
saxpy -- -1.0 pat uno antipat

fmac antipat kSlice2 kSlice2_new
saxpy 1 kSlice kSlice2_new kSlice3

fft -i -u 6 kSlice3 coil_images3
fmac -C -s 8 coil_images3 map im3


