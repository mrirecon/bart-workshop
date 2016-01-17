#!/bin/bash

set -eu
set -o pipefail

set -x

TOOLBOX_PATH=${TOOLBOX_PATH?="TOOLBOX_PATH not set!"}

# Reconstruct with l1-wavelets in space and total variation in time
bart pics -i 100 -p weights -R T:$(bart bitmask 10):0:.02 -R W:$(bart bitmask 0 1 2):0:0.001 ksp maps recon_wavtv
bart slice 4 0 recon_wavtv tmp001 # throw out second ESPIRiT image

## FFT-interpolate
bart fft -u 7 tmp001 recon_wavtv
bart resize -c 1 80 recon_wavtv tmp001
bart fft -iu 7 tmp001 recon_wavtv

# Reconstruct with locally low rank across space and time
bart pics -i 100 -p weights -R L:$(bart bitmask 0 1 2):0:0.05 ksp maps recon_llr
bart slice 4 0 recon_llr tmp001 # throw out second ESPIRiT image

## FFT-interpolate
bart fft -u 7 tmp001 recon_llr
bart resize -c 1 80 recon_llr tmp001
bart fft -iu 7 tmp001 recon_llr

# Combine, and compare
bart join 1 recon_wavtv recon_llr recon_compare # combine the reconstructions

rm tmp001.cfl tmp001.hdr
