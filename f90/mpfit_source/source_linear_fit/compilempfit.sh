#!/bin/bash
# ifort produces a buggy code!!! must use gfortran
gfortran nrtype.f90 nrutil.f90 nr.f90 functions.f90 interface.f90 mpfit.f90   -o mpfit

