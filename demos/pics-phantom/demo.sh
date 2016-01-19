#!/bin/bash

set -eu
set -o pipefail

function handle_signal() {
    exit 1
}
trap handle_signal SIGABRT SIGTERM SIGINT SIGHUP SIGSEGV

TOOLBOX_PATH=${TOOLBOX_PATH?="TOOLBOX_PATH not set!"}
export PATH=${TOOLBOX_PATH}:${PATH}

VIEWER="bart toimg"
VIEWER=bartview.py
VIEWER=viewer
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
nmaps=8 # coil sensitivities
dim=256 # ky/kz dims

# make k-space and sensitivities separately
bart phantom -x $dim -k ksp
bart phantom -x $dim -S $nmaps maps_orig
bart fft -iu 7 ksp img_orig
bart fmac img maps_orig cimg_orig
bart fft -u 7 cimg_orig ksp_orig
rmcfl cimg_orig

# Add noise, based on SNR of k-space.
#bart ones 1 1 o # dummy variable for summation
#bart rss 0 ksp ksp_mag # take abs value of k-space
#bart fmac -s `bart bitmask 0 1 2 3 4 5` ksp_mag o ksp_sum
#noisevar=`bart show ksp_sum`
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
ncc=6 # coil compression
bart cc -P $ncc ksp_und ksp_und_cc

# ESPIRiT sensitivity map calibration
bart ecalib -S ksp_und_cc maps

# View the sensitivities
cflview maps &

# ESPIRiT reconstruction
l2=.01
bart pics -R Q:$l2 -S ksp_und_cc maps img_recon0
bart slice 4 0 img_recon0 recon_l2 # remove the extra map

# L1-ESPIRiT with Wavelets
l1wav=.008
bart pics -S -R W:7:0:$l1wav ksp_und_cc maps img_recon0
bart slice 4 0 img_recon0 recon_wav # remove the extra map

# L1-ESPIRiT with Total Variation
l1tv=.008
bart pics -S -R T:7:0:$l1tv ksp_und_cc maps img_recon0
bart slice 4 0 img_recon0 recon_tv # remove the extra map
rmcfl img_recon0

# Zero-fill reconstruction
bart fft -iu 7 ksp_und_cc cimg
bart rss 8 cimg img_zf
rmcfl cimg

# Compare the results
bart join 2 img_noise img_zf recon_l2 recon_wav recon_tv img_compare
cflview img_compare &
