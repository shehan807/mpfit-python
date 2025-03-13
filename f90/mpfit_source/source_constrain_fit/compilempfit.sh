#!/bin/bash
ifort nrtype.f90 nrutil.f90 nr.f90 brent.f90 mnbrak.f90 variables.f90 mpfitroutines.f90 linmin.f90 frprmn.f90 mpfit.f90  -o mpfit_constrain

