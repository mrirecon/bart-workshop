#!/bin/bash
set -e

SCALE=0.6
SPOKES=21
PHASES=100
KSPACE=data/slice
SENS=data/sens

READ=$(bart show -d0 $KSPACE)
LINES=$(bart show -d1 $KSPACE)
PHASES=$(($LINES / $SPOKES))

# create Golden-ratio radial trajectory
bart traj -G -x$READ -y$LINES tmp1

# over-sampling x2
bart scale $SCALE tmp1 tmp2

# split off time dimension into index 10
bart reshape $(bart bitmask 2 10) $SPOKES $PHASES tmp2 traj

rm tmp1.* tmp2.*

# split-off time dim
bart reshape $(bart bitmask 1 2) $SPOKES $PHASES $KSPACE tmp1

# move time dimensions to dim 10 and reshape
bart transpose 2 10 tmp1 tmp2 

bart reshape $(bart bitmask 0 1 2) 1 $READ $SPOKES tmp2 kspace

rm tmp1.* tmp2.*

bart pics -S -d5 -u10. -i100 -RT:$(bart bitmask 10):0:0.01 -i50 -t traj kspace $SENS img

rm traj.* kspace.*

