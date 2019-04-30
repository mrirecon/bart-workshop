BUILDTYPE = Linux
UNAME = $(shell uname -s)

ifeq ($(UNAME),Darwin)
    BUILDTYPE = MacOSX
endif

ifeq ($(TOOLBOX_PATH),)
TOOLBOX_INC=/usr/include/bart/
TOOLBOX_LIB=/usr/lib/bart/
else
TOOLBOX_INC=$(TOOLBOX_PATH)/src/
TOOLBOX_LIB=$(TOOLBOX_PATH)/lib/
endif


CC ?= gcc
CFLAGS ?= -Wall -O2
CFLAGS += -std=c11 -fopenmp


ifneq ($(BUILDTYPE), MacOSX)
BLAS_BASE ?= /usr/
else
ifeq ($(MACPORTS),1)
BLAS_BASE ?= /opt/local/
CPPFLAGS += -DUSE_MACPORTS
endif
BLAS_BASE ?= /usr/local/opt/openblas/
endif


ifeq ($(BUILDTYPE), MacOSX)
    EXPDYN = -Wl,-export_dynamic
else
    EXPDYN = -export-dynamic
endif

-include Makefile.local


all: sense_recon

sense_recon:	src/sense_recon.c 
	$(CC) $(CFLAGS) $(EXPDYN) -o sense_recon src/sense_recon.c -I$(TOOLBOX_INC) -L$(TOOLBOX_LIB) -liter -lnoncart -llinops -lsense -lnum -lmisc -lm -lfftw3f -lfftw3f_threads -lopenblas  -lm 

clean:
	rm -f sense_recon
