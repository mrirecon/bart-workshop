
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "iter/iter.h"

#include "noncart/nufft.h"

#include "linops/linop.h"

#include "sense/model.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"
#include "misc/mri.h"

#ifndef DIMS
#define DIMS 16
#endif



static const char usage_str[] = "<trajectory> <rawdata> <sens> <img>";
static const char help_str[] = "Perform non-Cartesian CG-SENSE reconstruction with l2 regularization.";



/* 
 * Returns a non-Cartesian SENSE forward model linear operator, consisting of sensitivity map operator
 * and NUFFT operator.
 */
static const struct linop_s* sense_nc_init(const long sens_dims[DIMS], const complex float* sens, const long ksp_dims[DIMS], const long traj_dims[DIMS], const complex float* traj, struct nufft_conf_s conf)
{
	long cimg_dims[DIMS]; // coil image dimensions
	long img_dims[DIMS];

	// the coil dimensions are the same as the sens dimensions, except for the ESPIRiT maps dimension
	md_select_dims(DIMS, ~MAPS_FLAG, cimg_dims, sens_dims);

	// the image dimensions are the same as the sens dimensions, except for the coil dimension
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, sens_dims);


	/*
	 * We use the linop_s interface to create linear operators. There are many pre-defined operators in BART, and the 
	 * API makes it easy to add new linear operators
	 */

	// create the 3D NUFFT operator, F
	const struct linop_s* fft_op = nufft_create2(DIMS, ksp_dims, cimg_dims, traj_dims, traj, NULL, NULL, NULL, NULL, conf);

	// create the sensitivity maps operator, S
	const struct linop_s* maps_op = maps2_create(cimg_dims, sens_dims, img_dims, sens);

	/*
	* Chain the operators together: A = F S.
	*
	* Every linop has a linop->forward operator (Ax),
	* a linop->adjoint operator (A^H y), and
	* a linop->normal operator (A^H A x)
	*
	* By chaining two linops together, the operators are automatically
	* created. Many other linear algebra rules can be automatically
	* performed using the linop interface
	*/
	const struct linop_s* lop = linop_chain_FF(maps_op, fft_op);

	return lop;
}



int main(int argc, char* argv[])
{
	/*
	 * Create a default NUFFT operator config
	 * and use Toeplitz/low memory mode.
	 */
	struct nufft_conf_s nuconf = nufft_conf_defaults;
	nuconf.toeplitz = true;
	nuconf.lowmem = true;

	/*
	 * Create a default Conjugate Gradient Algorithm config.
	 * We will allow the user to modify its parameters
	 */
	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;

	/*
	 * Add command-line options to control the reconstruction. 
	 * Here we directly assign the options to the CG config,
	 * though these can be assigned to any variable.
	 */
	const struct opt_s opts[] = {

		OPT_UINT('i', &cgconf.maxiter, "itr", "maximum number of iterations"),
		OPT_FLOAT('r', &cgconf.l2lambda, "lambda", "l2 regularization parameter"),
	};

	/*
	 * Construct a command-line interface that requires four input arguments
	 * and takes the optional arguments defined above
	 */
	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	/* 
	 * Initialize the OpenMP threads, FFTW threads, etc.
	 */
	num_init();

	/* 
	 * All data dimensions are of size DIMS. By convention, the dimensions are
	 * [X, Y, Z, C, M, ...],
	 * where (X, Y, Z) is the spatial matrix size,
	 * C is the number of coils, and
	 * M is the number of ESPIRiT maps.
	 *
	 * Higher dimensions can be used for phases, echoes, etc.
	 */

	long traj_dims[DIMS];	// placeholder for trajectory dims
	long ksp_dims[DIMS];	// placeholder for kspace dims
	long sens_dims[DIMS];	// placeholder for sensitivity map dims
	long img_dims[DIMS];	// placeholder for output image dims


	/*
	 * Load the data into memory-mapped files
	 */
	complex float* traj = load_cfl(argv[1], DIMS, traj_dims);
	complex float* ksp = load_cfl(argv[2], DIMS, ksp_dims);
	complex float* sens = load_cfl(argv[3], DIMS, sens_dims);

	/* 
	 * The reconstructed image dimensions are [X, Y, Z, 1, M].
	 * We can select these dimensions using the bitmask notation,
	 * where "FFT_FLAGS = 7" selects dimensions (0, 1, 2)
	 * and "MAPS_FLAG = 16" selects dimension 4
	 */
	md_select_dims(DIMS, FFT_FLAGS | MAPS_FLAG, img_dims, sens_dims);

	debug_printf(DP_INFO, "Non-Cartesian CG-SENSE Reconstruction\n");

	/*
	 * Create the output file and initialize it to zeros
	 */
	complex float* img = create_cfl(argv[4], DIMS, img_dims);
	md_clear(DIMS, img_dims, img, CFL_SIZE);


	/* 
	 * Create the Non-Cartesian SENSE operator, A.
	 * This is a custom function defined above that creates a NUFFT op, a maps op,
	 * and chains them together.
	 */
	const struct linop_s* forward_op = sense_nc_init(sens_dims, sens, ksp_dims, traj_dims, traj, nuconf);

	/*
	 * Create storage for and compute the adjoint, A^H y.
	 * This will be used as the input to the CG algorithm.
	 * The linop_adjoint function calls the adjoint operation of our
	 * linear operator
	 */
	complex float* adj = md_alloc(DIMS, img_dims, CFL_SIZE);
	linop_adjoint(forward_op, DIMS, img_dims, adj, DIMS, ksp_dims, ksp);


	/* 
	 * The CG algorithm operates on real-valued floats,
	 * where the complex-valued arithmetic is handled by the linop interface.
	 * Therefore, we double the size of the problem and cast the variables to floats
	 */
	unsigned long size = 2 * md_calc_size(DIMS, img_dims);

	/*
	 * Call the CG iterative algorithm. We use the simplest interface,
	 * though we could also use the iter2_conjgrad interface.
	 * We directly pass to CG the normal equation operator 
	 */
	iter_conjgrad(CAST_UP(&cgconf), forward_op->normal, NULL, size, (float*)img, (float*)adj, NULL); 

	/*
	 * Free temporary memory and close the memory-mapped files
	 */

	md_free(adj);
	linop_free(forward_op);

	unmap_cfl(DIMS, traj_dims, traj);
	unmap_cfl(DIMS, ksp_dims, ksp);
	unmap_cfl(DIMS, sens_dims, sens);
	unmap_cfl(DIMS, img_dims, img);

	return 0;
}




