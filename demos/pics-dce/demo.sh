#!/bin/bash

set -eu
set -o pipefail


function handle_signal() {
    exit 1
}
trap handle_signal SIGABRT SIGTERM SIGINT SIGHUP SIGSEGV



# add the BART path to the system path
TOOLBOX_PATH=${TOOLBOX_PATH?="TOOLBOX_PATH not set!"}
export PATH=${TOOLBOX_PATH}:${PATH}

# simple function for viewing/saving images
#VIEWER="bart toimg"
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


# Interpolate by FFT zero-padding
function fft_interp () {
	bart fft -u 7 $2 $2.tmp
	bart resize -c 1 $1 $2.tmp $3.tmp
	bart fft -iu 7 $3.tmp $3
	
	rmcfl $2.tmp
	rmcfl $3.tmp
}


data_dir=data
ksp=${data_dir}/ksp
maps=${data_dir}/maps
weights=${data_dir}/weights
pat=${data_dir}/pat

bart show -m $ksp

# Compare the sampling pattern with the sample weights mask
bart pattern $ksp $pat # generate a sampling pattern from k-space
bart join 2 $pat $weights pat_weights
bart show -m pat_weights

# Sum along time
bart ones 1 1 o
bart fmac -s $(bart bitmask 10) pat_weights o pat_weights_sum
rmcfl o
cflview pat_weights_sum & # visualize the result

# Reconstruct with l1-wavelets in space and total variation in time
bart pics -i 100 -p $weights -R T:$(bart bitmask 10):0:.02 -R W:$(bart bitmask 0 1 2):0:0.001 $ksp $maps recon_wavtv
bart slice 4 0 recon_wavtv tmp001 # throw out second ESPIRiT image

## FFT-interpolate
fft_interp 80 tmp001 recon_wavtv

# Reconstruct with locally low rank across space and time
bart pics -i 100 -p $weights -R L:$(bart bitmask 0 1 2):$(bart bitmask 0 1 2):0.05 $ksp $maps recon_llr
bart slice 4 0 recon_llr tmp001 # throw out second ESPIRiT image

## FFT-interpolate
fft_interp 80 tmp001 recon_llr

# Combine and compare
bart join 1 recon_wavtv recon_llr recon_compare # combine the reconstructions

rmcfl tmp001

cflview recon_compare &
