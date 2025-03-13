!************************************
! this is from numerical recipes, and is used by
! the function minimization subroutine frprmn
!************************************

MODULE f1dim_mod
	USE nrtype
	INTEGER(I4B) :: ncom
        real*8,dimension(:),pointer::pcom
	REAL(SP), DIMENSION(:), POINTER :: xicom
CONTAINS
!BL
	FUNCTION f1dim(x)
	IMPLICIT NONE
	REAL(SP), INTENT(IN) :: x
	REAL(SP) :: f1dim
        integer::i,j
	INTERFACE
           FUNCTION kaisq(x)
             use variables
             real*8,dimension(:), INTENT(IN) :: x
             REAL*8 :: kaisq
		END FUNCTION kaisq
	END INTERFACE
        real*8,dimension(size(pcom))::xt
	xt=pcom+(x*xicom)
     	f1dim=kaisq(xt)
!!$        write(*,*) f1dim
      	END FUNCTION f1dim
END MODULE f1dim_mod


	SUBROUTINE linmin(p,xi,fret)
	USE nrtype; USE nrutil, ONLY : assert_eq
	USE nr, ONLY : mnbrak,brent
	USE f1dim_mod
	IMPLICIT NONE
	REAL(SP), INTENT(OUT) :: fret
        real*8,dimension(:),target,intent(inout)::p
	REAL(SP), DIMENSION(:), TARGET, INTENT(INOUT) :: xi
	REAL(SP), PARAMETER :: TOL=1.0e-4_sp
	REAL(SP) :: ax,bx,fa,fb,fx,xmin,xx
        integer::i
	ncom=size(xi)
	pcom=>p
	xicom=>xi
	ax=0.0
	xx=1.0
	call mnbrak(ax,xx,bx,fa,fx,fb,f1dim)
	fret=brent(ax,xx,bx,f1dim,TOL,xmin)
	xi=xmin*xi
	p=p+xi
	END SUBROUTINE linmin
