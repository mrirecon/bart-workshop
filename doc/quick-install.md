## Quick Installation
The quick-install instructions should be enough to get started for Linux or Mac OS X.
See the [README](https://github.com/mrirecon/bart/blob/master/README) for full installation instructions,
including Windows support.

These instructions are for [BART version 0.3.00](https://github.com/mrirecon/bart/releases/tag/v0.3.00). Newer versions
of BART may use slightly different syntax, e.g. `CUDA_BASE` instead of `cuda.top`

### Install the required dependencies.
For Linux,
```bash
sudo apt-get install gcc make libfftw3-dev liblapack-dev libpng-dev
```

For Mac OS X, using macports,
```bash
sudo port install fftw-3-single gcc47 libpng
```

Optionally, install [CUDA](https://developer.nvidia.com/cuda-downloads) and
[ACML](http://developer.amd.com/tools-and-sdks/archive/amd-core-math-library-acml/acml-downloads-resources/). Note ACML
is not supported under Mac OS X.

### Configure local Makefile paths
Make a local Makefile to configure paths specific to your system. Change these defaults as desired.
```bash
echo "PARALLEL=1 # speed up build" > Makefile.local
echo "CUDA=1 # compile with CUDA support" >> Makefile.local
echo "cuda.top := /usr/local/cuda" >> Makefile.local
```

For Linux,
```bash
echo "ACML=1 # compile with ACML support" >> Makefile.local
echo "acml.top := /usr/local/acml5.3.1/gfortran64_mp_int64" >> Makefile.local
```

For Mac,
```bash
echo "ACML=0 # No ACML support on Mac" >> Makefile.local
echo "fftw.top := /opt/local/" >> Makefile.local
```

### Build the source code
Under the `bart/` directory, run
```bash
make
```
This will build the binary `bart`, which is the entry point to the command-line tools.
```bash
./bart
#BART. Available commands are:
#bench       bitmask     bpsense     caldir      calmat      cc
#cdf97       circshift   conj        conv        cpyphs      creal
#crop        ecalib      ecaltwo     estdims     estvar      extract
#fakeksp     fft         fftmod      fftshift    filter      flip
#fmac        homodyne    itsense     join        lrmatrix    mip
#nlinv       noise       normalize   nrmse       nufft       ones
#pattern     phantom     pics        pocsense    poisson     repmat
#reshape     resize      rof         rsense      rss         sake
#saxpy       scale       sdot        show        slice       spow
#svd         threshold   toimg       traj        transpose   twixread
#version     walsh       wave        zeros
```

### Add BART to the path 
Under the `bart/` directory, run
```bash
source vars.sh
```
This adds the `bart` tool and the `TOOLBOX_PATH` environment variable to your session,
To add this for future sessions, append this line to your `.bashrc`.

### Test
Test the installation by benchmarking the toolbox:
```bash
bart bench
#                add (md_zaxpy) | 0.1157 0.1189 0.1170 0.1160 0.1169 | Avg: 0.1169 Max: 0.1189 Min: 0.1157
#    add (md_zaxpy), contiguous | 0.0727 0.0705 0.0700 0.0711 0.0703 | Avg: 0.0709 Max: 0.0727 Min: 0.0700
#                add (for loop) | 0.0404 0.0460 0.0431 0.0423 0.0424 | Avg: 0.0428 Max: 0.0460 Min: 0.0404
#                sum (md_zaxpy) | 0.1252 0.1211 0.1190 0.1181 0.1207 | Avg: 0.1208 Max: 0.1252 Min: 0.1181
#    sum (md_zaxpy), contiguous | 0.0829 0.0810 0.0857 0.0811 0.0927 | Avg: 0.0847 Max: 0.0927 Min: 0.0810
#                sum (for loop) | 0.0421 0.0423 0.0366 0.0392 0.0394 | Avg: 0.0399 Max: 0.0423 Min: 0.0366
#             complex transpose | 0.0583 0.0552 0.0602 0.0586 0.0634 | Avg: 0.0592 Max: 0.0634 Min: 0.0552
#                complex resize | 0.0019 0.0020 0.0021 0.0021 0.0020 | Avg: 0.0020 Max: 0.0021 Min: 0.0019
#       complex matrix multiply | 0.0554 0.0646 0.0562 0.0562 0.0535 | Avg: 0.0572 Max: 0.0646 Min: 0.0535
#       batch matrix multiply 1 | 0.0154 0.0141 0.0156 0.0142 0.0141 | Avg: 0.0147 Max: 0.0156 Min: 0.0141
#       batch matrix multiply 2 | 0.2275 0.2258 0.2275 0.2262 0.2271 | Avg: 0.2268 Max: 0.2275 Min: 0.2258
#        tall matrix multiply 1 | 0.0870 0.0875 0.0969 0.0901 0.0912 | Avg: 0.0905 Max: 0.0969 Min: 0.0870
#        tall matrix multiply 2 | 0.0231 0.0199 0.0206 0.0260 0.0249 | Avg: 0.0229 Max: 0.0260 Min: 0.0199
#           complex dot product | 0.0135 0.0136 0.0134 0.0137 0.0170 | Avg: 0.0142 Max: 0.0170 Min: 0.0134
#           complex dot product | 0.0138 0.0138 0.0133 0.0135 0.0188 | Avg: 0.0146 Max: 0.0188 Min: 0.0133
#      real complex dot product | 0.0012 0.0013 0.0012 0.0012 0.0013 | Avg: 0.0012 Max: 0.0013 Min: 0.0012
#                       l2 norm | 0.0011 0.0010 0.0011 0.0010 0.0015 | Avg: 0.0011 Max: 0.0015 Min: 0.0010
#                       l1 norm | 0.0154 0.0158 0.0163 0.0160 0.0138 | Avg: 0.0155 Max: 0.0163 Min: 0.0138
#                        copy 1 | 0.0182 0.0050 0.0059 0.0050 0.0050 | Avg: 0.0078 Max: 0.0182 Min: 0.0050
#                        copy 2 | 0.0051 0.0052 0.0050 0.0055 0.0053 | Avg: 0.0052 Max: 0.0055 Min: 0.0050
#           wavelet soft thresh | 0.0551 0.0546 0.0559 0.0600 0.0588 | Avg: 0.0569 Max: 0.0600 Min: 0.0546
#           wavelet soft thresh | 0.0797 0.0643 0.0635 0.0617 0.0590 | Avg: 0.0656 Max: 0.0797 Min: 0.0590
```
