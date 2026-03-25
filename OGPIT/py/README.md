OGPIT - Optimization by Gaussian Processes in Trust-regions
===========================================================

Contents
--------
* ``OGPIT_hetGPy.py`` - Main OGPIT functionality.
* ``OGPIT_hetGPy_cutest.py`` - OGPIT for the CUTEst benchmark 2 test function.
   - Users that would like to run this must first manually install
     [PyCUTEst](https://jfowkes.github.io/pycutest/_build/html/index.html)
     (including CUTEst).
   - This can be executed with MPI parallelization using `mpi4py`.
   - If parallel execution will be used, please execute once first in serial mode
     to generate a cache of `CUTEst` test problem definitions.
* ``local_benchPy.py`` - Run OGPIT on benchmark 1.
   - This can be executed with MPI parallelization using `mpi4py`.
* ``local_benchPy_bendfo.py`` - Run OGPIT on benchmark 2.
   - This can be executed with MPI parallelization using `mpi4py`.
* ``local_benchPy_BOtorch_bendfo.py`` - Run OGPIT with BoTorch and Turbo on benchmark 2.
   - This can be executed with MPI parallelization using `mpi4py`.
* ``make_data_profs.py`` - Postprocessing of benchmark results.
   - Users should manually alter settings in the file to specify which benchmarks to visualize.
* ``plot_data_profile.py`` - General code used by ``make_data_profs.py``.
* ``make_pyCUTEst_problem_list.py`` - General code used by ``OGPIT_hetGPy_cutest.py`` to genereate the CUTEst test problem set.
