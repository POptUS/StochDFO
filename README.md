# StochDFO
A Collection of Algorithms for Stochastic Derivative-Free Optimization.
Part of [POptUS: Practical Optimization Using Structure](https://github.com/POptUS).

### Repository

**TODO**: Add badges here

## License & Copyright

All code included in StochDFO is open source, with the particular form of
license contained in the top-level subdirectories.  If such a subdirectory does
not contain a LICENSE file, then it is automatically licensed as described in
the otherwise encompassing StochDFO [LICENSE](/LICENSE).

Copyright (c) 2026, Mickaël Binois and UChicago Argonne LLC through Argonne
National Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.

## Support

To 

* report potential problems with StochDFO,
* propose a change, or
* request a new feature,

please check if a related [Issue](https://github.com/POptUS/StochDFO/issues)
already exists before creating a new issue.  For all other communication, please
send an email to the POptUS development team at

 * ``poptus@mcs.anl.gov``

## Documentation

**TODO**: Determine what type of documentation should be built up & detail here.

## Installation & Testing

This repository presently contains only the OGPIT tool as a collection of Python
source code files.  To run OGPIT, it is suggested that users either

* run the code from ``/path/to/StochDFO/OGPIT/py`` or
* add ``/path/to/StochDFO/OGPIT/py`` to their ``PYTHONPATH`` environment
  variable.

Users can setup their target Python virtual environment with most of the OGPIT
external dependences by, for example, activating their environment and running
```
python -m pip install -r /path/to/StochDFO/requirements.txt
```

Some of the code requires the use of the contents of the
[BenDFO](https://github.com/POptUS/BenDFO) repository, which is not distributed
as a Python package.  After cloning the BenDFO repository and before using
OGPIT, users should set the ``BENDFO_PATH`` environment variable to the root of
their BenDFO clone.

## Contributing to StochDFO

**TODO**: Define this.

## Cite StochDFO

**TODO**: Define this once this software is available in earnest.  This should
likely be a package-related document (e.g., User Guide) listing all authors.
