MODULE interfac
  USE nrtype; USE nrutil

  INTERFACE
     SUBROUTINE Amat(nsite,xyzmult,xyzcharge,r1,r2,maxl,A)

       REAL*8,DIMENSION(:,:),INTENT(in)::xyzmult,xyzcharge
       REAL*8,INTENT(in)::r1,r2
       INTEGER,INTENT(in)::maxl,nsite
       REAL*8,DIMENSION(:,:),INTENT(out)::A

       INTERFACE
          FUNCTION RSH(l,m,cs,x,y,z)
            REAL*8::RSH
            INTEGER,INTENT(in)::l,m,cs
            REAL*8,INTENT(in)::x,y,z
          END FUNCTION RSH
       END INTERFACE

     END SUBROUTINE Amat
  END INTERFACE

  INTERFACE
     SUBROUTINE bvec(nsite,xyzmult,xyzcharge,r1,r2,maxl,multipoles,b)

       REAL*8,DIMENSION(:,:),INTENT(in)::xyzmult,xyzcharge
       REAL*8,INTENT(in)::r1,r2
       INTEGER,INTENT(in)::maxl,nsite
       REAL*8,DIMENSION(:,0:,0:,0:),INTENT(in)::multipoles
       REAL*8,DIMENSION(:),INTENT(out)::b

       INTERFACE
          FUNCTION RSH(l,m,cs,x,y,z)
            REAL*8::RSH
            INTEGER,INTENT(in)::l,m,cs
            REAL*8,INTENT(in)::x,y,z
          END FUNCTION RSH
       END INTERFACE
     END SUBROUTINE bvec
  END INTERFACE

  INTERFACE

     FUNCTION RSH(l,m,cs,x,y,z)
       REAL*8::RSH
       INTEGER,INTENT(in)::l,m,cs
       REAL*8,INTENT(in)::x,y,z
     END FUNCTION RSH

  END INTERFACE

  INTERFACE

     SUBROUTINE numbersites(inpfile,n)
       CHARACTER(30),INTENT(in)::inpfile
       INTEGER,INTENT(out)::n
     END SUBROUTINE numbersites

  END INTERFACE

  INTERFACE

     SUBROUTINE getmultmoments(inpfile,n,lmax,mm,ms,atomtype)
       CHARACTER(30),INTENT(in)::inpfile
       INTEGER,INTENT(in)::n
       integer,dimension(:),intent(out)::lmax
       REAL*8,DIMENSION(:,0:,0:,0:),INTENT(out)::mm
       REAL*8,DIMENSION(:,:),INTENT(out)::ms
       character(2),dimension(:),intent(out)::atomtype
     END SUBROUTINE getmultmoments

  END INTERFACE

  INTERFACE

     SUBROUTINE gencharges(ms,qs,midbond)
       REAL*8,DIMENSION(:,:),INTENT(in)::ms
       REAL*8,DIMENSION(:,:),INTENT(out)::qs
       INTEGER,DIMENSION(:,:),INTENT(in)::midbond
     END SUBROUTINE gencharges

  END INTERFACE

  INTERFACE

     SUBROUTINE svdcmp_sp(a,w,v)
       USE nrtype; USE nrutil, ONLY : assert_eq,nrerror,outerprod
       USE nr, ONLY : pythag
       IMPLICIT NONE
       REAL(SP), DIMENSION(:,:), INTENT(INOUT) :: a
       REAL(SP), DIMENSION(:), INTENT(OUT) :: w
       REAL(SP), DIMENSION(:,:), INTENT(OUT) :: v
     END SUBROUTINE svdcmp_sp

  END INTERFACE

  INTERFACE

     SUBROUTINE svbksb_sp(u,w,v,b,x)
       USE nrtype; USE nrutil, ONLY : assert_eq
       REAL(SP), DIMENSION(:,:), INTENT(IN) :: u,v
       REAL(SP), DIMENSION(:), INTENT(IN) :: w,b
       REAL(SP), DIMENSION(:), INTENT(OUT) :: x
     END SUBROUTINE svbksb_sp

  END INTERFACE


END MODULE interfac
      

