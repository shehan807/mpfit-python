!***************************
! this is a module of global variables used by
! subroutines in the mpfit program
!***************************

module variables

  !**************************************
  !maxl is the highest l value of all the sites (rank of largest multipole)
  ! currently, this cannot be greater than 4
  Integer,parameter::maxl=4  
  !**************************************

  !**************************************
  ! this parameters, control the hypersurface over which the electrostatic potential
  ! will be fit for each multipole distribution.
  ! the charges will therefore be sensitive to the choice of these parameters!!!
  ! currently, using values from the literature ...
  !REAL*8,PARAMETER::r1=2.27,r2=5.67      !values from JCC,12,(1991), 913-917
  REAL*8,PARAMETER::r1=3.78,r2=9.45        !values from JPC 1993,97,6628
  !**************************************

  !**********************************************************
  ! these parameters are used to constrain the total charge on the molecule.
  ! this is never really necessary, as the correct charge on the molecule
  ! "falls out" automatically from the fit. 
  ! if parameter "conchg" is set to 0, this does nothing.
  REAL*8,PARAMETER::molecule_charge=0.0,conchg=0d0
  !************************************************************


  INTEGER,DIMENSION(:),ALLOCATABLE::lmax
  real*8,dimension(:),allocatable::qstore
  REAL*8,DIMENSION(:,:),ALLOCATABLE::xyzcharge
  REAL*8,DIMENSION(:,:),ALLOCATABLE::xyzmult,allcharge
  REAL*8,DIMENSION(:,:,:,:),ALLOCATABLE::multipoles
  character(3),dimension(:),allocatable::atomtype,atmtype
  integer,dimension(:,:),allocatable::quse
  real*8,dimension(:),allocatable::rmin,rvdw
  REAL*8, PARAMETER :: PI=3.14159265359
  real*8,parameter::marker=500.
  Integer::site,countatom,countmid

end module variables
