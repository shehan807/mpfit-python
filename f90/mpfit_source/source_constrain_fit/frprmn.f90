 !**************************************
 ! this is a conjugate gradient function minimization subroutine
 ! that is adapted from numerical recipes
 !**************************************
	SUBROUTINE frprmn(p,ftol,iter,fret)
	USE nrtype; USE nrutil, ONLY : nrerror
	USE nr, ONLY : linmin
	IMPLICIT NONE
	INTEGER(I4B), INTENT(OUT) :: iter
	REAL(SP), INTENT(IN) :: ftol
	REAL(SP), INTENT(OUT) :: fret
	real*8,dimension(:), INTENT(INOUT) :: p
	INTERFACE
		FUNCTION kaisq(p)
                  use variables
		IMPLICIT NONE
		real*8,dimension(:), INTENT(IN) :: p
		REAL*8 :: kaisq
		END FUNCTION kaisq
!BL
		FUNCTION dkaisq(p)
                  use variables
                  IMPLICIT NONE
                  real*8,dimension(:), INTENT(IN) :: p
                  REAL*8, DIMENSION(size(p)) :: dkaisq
		END FUNCTION dkaisq
	END INTERFACE
	INTEGER(I4B), PARAMETER :: ITMAX=10000
	REAL(SP), PARAMETER :: EPS=1.0e-10_sp
	INTEGER(I4B) :: its
	REAL(SP) :: dgg,fp,gam,gg
	REAL(SP), DIMENSION(size(p)) :: g,h,xi
	fp=kaisq(p)
	xi=dkaisq(p)
       	g=-xi
	h=g
	xi=h
	do its=1,ITMAX
		iter=its
		call linmin(p,xi,fret)
		if (2.0_sp*abs(fret-fp) <= ftol*(abs(fret)+abs(fp)+EPS)) RETURN
		fp=fret
		xi=dkaisq(p)
		gg=dot_product(g,g)
!		dgg=dot_product(xi,xi)
		dgg=dot_product(xi+g,xi)
		if (gg == 0.0) RETURN
		gam=dgg/gg
		g=-xi
		h=g+gam*h
		xi=h
	end do
	call nrerror('frprmn: maximum iterations exceeded')
	END SUBROUTINE frprmn
