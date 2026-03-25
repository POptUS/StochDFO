OGPIT - Optimization by Gaussian Processes in Trust-regions
===========================================================

Contents
--------
* ``OGPIT_hetGPy.py`` - Main OGPIT functionality.
* ``OGPIT_hetGPy_cutest.py`` - OGPIT for the CUTEst benchmark 2 test function.
   - Users that would like to run this must also manually install
     [PyCUTEst](https://jfowkes.github.io/pycutest/_build/html/index.html)
     (including CUTEst).
   - If parallel execution with MPI is desired, please execute once first in serial mode
     to generate a cached of `CUTEst` test problem definitions
* ``local_benchPy.py`` - Run OGPIT on benchmark 1.
* ``local_benchPy_bendfo.py`` - Run OGPIT on benchmark 2.
* ``local_benchPy_BOtorch_bendfo.py`` - Run OGPIT with BoTorch and Turbo on benchmark 2.
* ``make_pyCUTEst_problem_list.py`` - List the test problems in the CUTEst problem set.
* ``make_data_profs.py`` - postprocessing of benchmark results.
   - Users should manually alter settings in the file to specify which benchmarks to visualize.
* ``plot_data_profile.py`` - general code used by `make_data_profs.py``
