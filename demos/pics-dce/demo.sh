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

function fft_interp () {
	bart fft -u 7 $1 $1.tmp
	bart resize -c 1 80 $1.tmp $2.tmp
	bart fft -iu 7 $2.tmp $2
	
	rmcfl $1.tmp
	rmcfl $2.tmp
}

TOOLBOX_PATH=${TOOLBOX_PATH?="TOOLBOX_PATH not set!"}
export PATH=${TOOLBOX_PATH}:${PATH}

data_dir=data
ksp=${data_dir}/ksp
maps=${data_dir}/maps
weights=${data_dir}/weights

# Reconstruct with l1-wavelets in space and total variation in time
bart pics -i 100 -p $weights -R T:$(bart bitmask 10):0:.02 -R W:$(bart bitmask 0 1 2):0:0.001 $ksp $maps recon_wavtv
bart slice 4 0 recon_wavtv tmp001 # throw out second ESPIRiT image

## FFT-interpolate
fft_interp tmp001 recon_wavtv

# Reconstruct with locally low rank across space and time
bart pics -i 100 -p $weights -R L:$(bart bitmask 0 1 2):0:0.05 $ksp $maps recon_llr
bart slice 4 0 recon_llr tmp001 # throw out second ESPIRiT image

## FFT-interpolate
fft_interp tmp001 recon_llr

# Combine and compare
bart join 1 recon_wavtv recon_llr recon_compare # combine the reconstructions

rmcfl tmp001

cflview recon_compare &
