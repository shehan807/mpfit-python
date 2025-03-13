!*********************************************************
!  This program fits points charges to a distributed multipole expansion (DMA)
!  as calculated by the algorithm described in
!  Stone, A.J. Chem.Phys.Lett. 1981, 83, 233-239
!  and
!  Stone,A.J.;Alderton,M. Mol. Phys. 1985, 56, 1047-1064
!  which is implemented in MOLPRO, or other software
!  See:  http://www-stone.ch.cam.ac.uk/programs/
!  
!  point charges are fit to each individual multipole distribution
!  on every atom, by best reproducing the electrostatic potential of that 
!  specific multipole distribution.  The total charge on every atom is then the
!  sum of the charges on the atom from each of these separate fits.
!
!  Atoms that share the same label, are treated as identical atom types, and are
!  constrained to give the same total charge
!
!  all units in this program are in a.u.
!
!  this code is based on algorithm described in 
!  Ferenczy,G.G., J. Comput. Chem. 1991, 12, 913-917
!  see also
!  JPC 1993,97,6628
!*********************************************************

PROGRAM mpfit
  use variables
  USE mpfitroutines
  REAL*8,PARAMETER::small=1.0D-7,ftol=1D-15
  CHARACTER(50)::inpfile,chargefile
  INTEGER::multsites,chargesites,i,j,k,qsites,printt,count,nmdbnd,atoms,mid,numparam,iter,atms
  REAL*8::rqm,f0,f1,fdel,fret,molchg,sum
  REAL*8,DIMENSION(:),ALLOCATABLE::p0,p1,delta,df0,q
  interface
     function kaisq(p0)
       use variables
       real*8::kaisq
       real*8,dimension(:),intent(in)::p0
     end function kaisq
     function dkaisq(p0)
       use variables
       real*8,dimension(:),intent(in)::p0
       real*8,dimension(size(p0))::dkaisq
     end function dkaisq
     subroutine expandcharge(p0)
       real*8,dimension(:),intent(in)::p0
     end subroutine expandcharge
     SUBROUTINE frprmn(p,ftol,iter,fret)
       USE nrtype; USE nrutil, ONLY : nrerror
       USE nr, ONLY : linmin
       IMPLICIT NONE
       INTEGER(I4B), INTENT(OUT) :: iter
       REAL(SP), INTENT(IN) :: ftol
       REAL(SP), INTENT(OUT) :: fret
       real*8,dimension(:), INTENT(INOUT) :: p
     end subroutine frprmn
  end interface

  CALL getarg(1,inpfile)
  CALL numbersites(inpfile,multsites)


  !**************************** WARN about units *******************************
  !write(*,*) ""
  !write(*,*) "NOTE :: all input should be in atomic units including coordinates (Bohr)"
  !write(*,*) "and multipole moments.  The output charges are in atomic units"
  !write(*,*) ""
  !*****************************************************************************


  ALLOCATE(multipoles(multsites,0:maxl,0:maxl,0:1),xyzmult(multsites,3))
  allocate(lmax(multsites),rvdw(multsites),atomtype(multsites))

  CALL getmultmoments(inpfile,multsites,lmax,multipoles,xyzmult,atomtype)
  chargesites=multsites
  ALLOCATE(xyzcharge(chargesites,3),qstore(chargesites),quse(multsites,chargesites))
  allocate(allcharge(multsites,chargesites))


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  CALL gencharges(xyzmult,xyzcharge)
  call genvdw(rvdw,atomtype)
  call numindatommid(atomtype,atms)
  allocate(atmtype(atms))
  call collapseatommid(atomtype,atmtype)


  numparam=0
  DO i=1,multsites
     site=i
     call sitespecificdata
     qsites=countatom
     numparam=numparam+qsites
  enddo

  numparam=numparam-(size(atomtype)-size(atmtype))
  allocate(p0(numparam),p1(numparam),df0(numparam),delta(numparam))

  p0=-0.001  
  ! this fills in charges based on the unique fitting parameters
  call expandcharge(p0)

 !  This is the fitting program
  call frprmn(p0,ftol,iter,fret)

 ! for the fit charges q, add them to the total array qstore
 ! qstore then holds the total charge for every atom
  call expandcharge(p0)

  ! total charge on molecule
  molchg=0.0
  do j=1,chargesites
     molchg=molchg+qstore(j)
  enddo

  DO j=1,multsites
      WRITE(*,'(A,7x,F8.5)') atomtype(j),qstore(j)
  ENDDO

END PROGRAM mpfit

       
      


