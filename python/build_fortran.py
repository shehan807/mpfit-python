import numpy as np
import os

# Build the Fortran module
os.system('f2py -c nrtype.f90 nrutil.f90 nr.f90 functions.f90 interface.f90 mpfit.f90 -m fortran_functions')
