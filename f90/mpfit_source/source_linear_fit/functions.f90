SUBROUTINE Amat(nsite,xyzmult,xyzcharge,r1,r2,maxl,A)
!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!this subroutine constructs the matrix as in
  !! J. Comp. Chem. Vol. 12, No. 8, 913-917 (1991)
  !! returns a 3 dimensional array A(i,j,k), where i stands for the specific
  !! multipole and j,k stand for the charges
!!!!!!!!!!!!!!!!!!!!!!!!!!!
  REAL*8,DIMENSION(:,:),INTENT(in)::xyzmult,xyzcharge
  REAL*8,INTENT(in)::r1,r2
  INTEGER,INTENT(in)::maxl,nsite
  REAL*8,DIMENSION(:,:),INTENT(out)::A
  REAL*8,DIMENSION(:),ALLOCATABLE::W
  INTEGER::i,j,k,l,m,ncharge
  REAL*8::sum,xj,yj,zj,xk,yk,zk
  INTERFACE
     FUNCTION RSH(l,m,cs,x,y,z)
       REAL*8::RSH
       INTEGER,INTENT(in)::l,m,cs
       REAL*8,INTENT(in)::x,y,z
     END FUNCTION RSH
  END INTERFACE

  ncharge=SIZE(xyzcharge(:,1))

  ALLOCATE(W(0:maxl))

  DO i=0,maxl
     W(i)=(1.0/(1.0-2.0*i))*(r2**(1-2*i)-r1**(1-2*i))
  ENDDO

     DO j=1, ncharge
        xj=xyzcharge(j,1)-xyzmult(nsite,1)
        yj=xyzcharge(j,2)-xyzmult(nsite,2)
        zj=xyzcharge(j,3)-xyzmult(nsite,3)
        DO k=1, ncharge
           xk=xyzcharge(k,1)-xyzmult(nsite,1)
           yk=xyzcharge(k,2)-xyzmult(nsite,2)
           zk=xyzcharge(k,3)-xyzmult(nsite,3)
           sum=0.0
           DO l=0,maxl
              IF(l .EQ. 0) THEN
                 sum=(1.0/(2.0*l+1.0))*W(0)*RSH(0,0,0,xj,yj,zj)*RSH(0,0,0,xk,yk,zk)
              ELSE
                 DO m=0,l
                    IF(m .EQ. 0) THEN
                       sum=sum+(1.0/(2.0*l+1.0))*W(l)*(RSH(l,0,0,xj,yj,zj)*RSH(l,0,0,xk,yk,zk))
                    ELSE
                       sum=sum+(1.0/(2.0*l+1.0))*W(l)*(RSH(l,m,0,xj,yj,zj)*RSH(l,m,0,xk,yk,zk)&
                            &+RSH(l,m,1,xj,yj,zj)*RSH(l,m,1,xk,yk,zk))
                    ENDIF
                 ENDDO
              ENDIF
           ENDDO
           A(j,k)=sum
        ENDDO
     ENDDO

END SUBROUTINE Amat

SUBROUTINE bvec(nsite,xyzmult,xyzcharge,r1,r2,maxl,multipoles,b)
!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!this subroutine constructs the vector as in
  !! J. Comp. Chem. Vol. 12, No. 8, 913-917 (1991)
  !! returns a 2 dimensional array b(i,k), where i stands for the specific
  !! multipole and k stands for the charge
!!!!!!!!!!!!!!!!!!!!!!!!!!!
  REAL*8,DIMENSION(:,:),INTENT(in)::xyzmult,xyzcharge
  REAL*8,INTENT(in)::r1,r2
  INTEGER,INTENT(in)::maxl,nsite
  REAL*8,DIMENSION(:,0:,0:,0:),INTENT(in)::multipoles
  REAL*8,DIMENSION(:),INTENT(out)::b
  REAL*8,DIMENSION(:),ALLOCATABLE::W
  INTEGER::i,j,k,l,m,ncharge
  REAL*8::sum,xk,yk,zk
  INTERFACE
     FUNCTION RSH(l,m,cs,x,y,z)
       REAL*8::RSH
       INTEGER,INTENT(in)::l,m,cs
       REAL*8,INTENT(in)::x,y,z
     END FUNCTION RSH
  END INTERFACE

  ncharge=SIZE(xyzcharge(:,1))

  ALLOCATE(W(0:maxl))

  DO i=0,maxl
     W(i)=(1.0/(1.0-2.0*i))*(r2**(1-2*i)-r1**(1-2*i))
  ENDDO

     DO k=1,ncharge
        xk=xyzcharge(k,1)-xyzmult(nsite,1)
        yk=xyzcharge(k,2)-xyzmult(nsite,2)
        zk=xyzcharge(k,3)-xyzmult(nsite,3)
        sum=0.0
        DO l=0,maxl
           IF(l .EQ. 0) THEN
              sum=(1.0/(2.0*l+1.0))*W(0)*multipoles(nsite,0,0,0)*RSH(0,0,0,xk,yk,zk)
           ELSE
              DO m=0,l
                 IF(m .EQ. 0) THEN
                    sum=sum+(1.0/(2.0*l+1.0))*W(l)*multipoles(nsite,l,0,0)*RSH(l,0,0,xk,yk,zk)
                 ELSE
                    sum=sum+(1.0/(2.0*l+1.0))*W(l)*(multipoles(nsite,l,m,0)*RSH(l,m,0,xk,yk,zk)&
                         &+multipoles(nsite,l,m,1)*RSH(l,m,1,xk,yk,zk))
                 ENDIF
              ENDDO
           ENDIF
        ENDDO
        b(k)=sum
     ENDDO

!!$     xk=xyzcharge(1,1)-xyzmult(nsite,1)
!!$     yk=xyzcharge(1,2)-xyzmult(nsite,2)
!!$     zk=xyzcharge(1,3)-xyzmult(nsite,3)
!!$write(*,*) RSH(0,0,0,xk,yk,zk)
!!$do l=1,maxl
!!$   write(*,*) RSH(l,0,0,xk,yk,zk),(RSH(l,m,0,xk,yk,zk),RSH(l,m,1,xk,yk,zk),m=1,l)
!!$enddo

END SUBROUTINE bvec


  FUNCTION RSH(l,m,cs,x,y,z)
!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!   this function evaluates regular spherical harmonics
    !!   at point x,y,z where l,m,cs determine the rank
    !!   cs=0 means cosine, cs=1 means sine
    !!  
!!!!!!!!!!!!!!!!!!!!!!!!!!!
    REAL*8::RSH
    INTEGER,INTENT(in)::l,m,cs
    REAL*8,INTENT(in)::x,y,z
    REAL*8::rsq
    REAL*8,DIMENSION(:,:,:),ALLOCATABLE::rsharray

    ALLOCATE(rsharray(0:4,0:4,0:1))
    rsq=x**2+y**2+z**2

    rsharray(0,0,0)=1.0
    rsharray(1,0,0)=z
    rsharray(1,1,0)=x
    rsharray(1,1,1)=y
    rsharray(2,0,0)=0.5*(3.0*z**2-rsq)
    rsharray(2,1,0)=SQRT(3.0)*x*z
    rsharray(2,1,1)=SQRT(3.0)*y*z
    rsharray(2,2,0)=0.5*SQRT(3.0)*(x**2-y**2)
    rsharray(2,2,1)=SQRT(3.0)*x*y
    rsharray(3,0,0)=0.5*(5.0*z**3-3.0*z*rsq)
    rsharray(3,1,0)=0.25*SQRT(6.0)*(4.0*x*z**2-x**3-x*y**2)
    rsharray(3,1,1)=0.25*SQRT(6.0)*(4.0*y*z**2-y*x**2-y**3)
    rsharray(3,2,0)=0.5*SQRT(15.0)*z*(x**2-y**2)
    rsharray(3,2,1)=SQRT(15.0)*x*y*z
    rsharray(3,3,0)=0.25*SQRT(10.0)*(x**3-3.0*x*y**2)
    rsharray(3,3,1)=0.25*SQRT(10.0)*(3.0*x**2*y-y**3)

    rsharray(4,0,0)=0.125*(8.0*z**4-24.0*(x**2+y**2)*z**2+3.0*(x**4+2.0*x**2*y**2+y**4))
    rsharray(4,1,0)=0.25*SQRT(10.0)*(4.0*x*z**3-3.0*x*z*(x**2+y**2))
    rsharray(4,1,1)=0.25*SQRT(10.0)*(4.0*y*z**3-3.0*y*z*(x**2+y**2))
    rsharray(4,2,0)=0.25*SQRT(5.0)*(x**2-y**2)*(6.0*z**2-x**2-y**2)
    rsharray(4,2,1)=0.25*SQRT(5.0)*x*y*(6.0*z**2-x**2-y**2)
    rsharray(4,3,0)=0.25*SQRT(70.0)*z*(x**3-3.0*x*y**2)
    rsharray(4,3,1)=0.25*SQRT(70.0)*z*(3.0*x**2*y-y**3)
    rsharray(4,4,0)=0.125*SQRT(35.0)*(x**4-6.0*x**2*y**2+y**4)
    rsharray(4,4,1)=0.125*SQRT(35.0)*x*y*(x**2-y**2)

    RSH=rsharray(l,m,cs)

  END FUNCTION RSH

  SUBROUTINE numbersites(inpfile,n)
    CHARACTER(*),INTENT(in)::inpfile
    INTEGER,INTENT(out)::n
    INTEGER::count,inputstatus,maxl,i
    real*8::x,y,z
    character::type
    CHARACTER(50)::line

count=0
    OPEN(unit=3,file=inpfile,status='old')
    DO 
       READ(3,*,Iostat=inputstatus) type,x,y,z,maxl
       IF (inputstatus < 0) EXIT
       count=count+1
       DO i=0, maxl
          READ(3,*) line
       ENDDO
    ENDDO
CLOSE(3)
n=count

  END SUBROUTINE numbersites

  SUBROUTINE getmultmoments(inpfile,n,lmax,mm,ms,atomtype)
    CHARACTER(*),INTENT(in)::inpfile
    INTEGER,INTENT(in)::n
    integer,dimension(:),intent(out)::lmax
    REAL*8,DIMENSION(:,0:,0:,0:),INTENT(out)::mm
    REAL*8,DIMENSION(:,:),INTENT(out)::ms
    character(2),dimension(:),intent(out)::atomtype
    INTEGER::i,l,num
    REAL*8::x,y,z,q0
    character(2)::type

    OPEN(unit=3,file=inpfile,status='old')

    DO i=1,n
       READ(3,*) atomtype(i),x,y,z,lmax(i)
       ms(i,1)=x;ms(i,2)=y;ms(i,3)=z
       READ(3,*) q0
       mm(i,0,0,0)=q0
       if(lmax(i) > 0) then
       DO l=1,lmax(i)
          READ(3,*) mm(i,l,0,0),(mm(i,l,j,0),mm(i,l,j,1),j=1,l)
       ENDDO
       endif
    ENDDO

    CLOSE(3)
!!$    DO i=1,n
!!$       WRITE(*,*) i,ms(i,1),ms(i,2),ms(i,3),lmax(i)
!!$       WRITE(*,*) mm(i,0,0,0)
!!$         if(lmax(i) > 0 ) then
!!$       DO l=1,lmax(i)
!!$          WRITE(*,*) mm(i,l,0,0),(mm(i,l,j,0),mm(i,l,j,1),j=1,l)
!!$       ENDDO
!!$    endif
!!$    ENDDO

  END SUBROUTINE getmultmoments

SUBROUTINE gencharges(ms,qs,midbond)
  REAL*8,DIMENSION(:,:),INTENT(in)::ms
  REAL*8,DIMENSION(:,:),INTENT(out)::qs
  INTEGER,DIMENSION(:,:),INTENT(in)::midbond
  INTEGER::i,j,nmult,nmid,count

nmult=SIZE(ms(:,1))
nmid=SIZE(qs(:,1))-SIZE(ms(:,1))

DO i=1,nmult
   qs(i,1)=ms(i,1)
   qs(i,2)=ms(i,2)
   qs(i,3)=ms(i,3)
ENDDO

IF (nmid > 0) THEN
   count=1
   DO i=1,nmult
      DO j=i+1,nmult
         IF (midbond(i,j) .EQ. 1) THEN
            qs(nmult+count,1)=(ms(i,1)+ms(j,1))/2.0
            qs(nmult+count,2)=(ms(i,2)+ms(j,2))/2.0
            qs(nmult+count,3)=(ms(i,3)+ms(j,3))/2.0
            count=count+1
          ENDIF
      ENDDO
    ENDDO
ENDIF

END SUBROUTINE gencharges

SUBROUTINE svdcmp_sp(a,w,v)
  USE nrtype; USE nrutil, ONLY : assert_eq,nrerror,outerprod
  IMPLICIT NONE
  REAL(SP), DIMENSION(:,:), INTENT(INOUT) :: a
  REAL(SP), DIMENSION(:), INTENT(OUT) :: w
  REAL(SP), DIMENSION(:,:), INTENT(OUT) :: v
  INTEGER(I4B) :: i,its,j,k,l,m,n,nm,iii
  REAL(SP) :: anorm,c,f,g,h,s,scale,x,y,z
  REAL(SP), DIMENSION(SIZE(a,1)) :: tempm
  REAL(SP), DIMENSION(SIZE(a,2)) :: rv1,tempn

  INTERFACE
     FUNCTION pythag(a,b)
       USE nrtype
       IMPLICIT NONE
       REAL(SP), INTENT(IN) :: a,b
       REAL(SP) :: pythag
     END FUNCTION pythag
  END INTERFACE


	m=SIZE(a,1)
	n=assert_eq(SIZE(a,2),SIZE(v,1),SIZE(v,2),SIZE(w),'svdcmp_sp')
	g=0.0
	scale=0.0
	DO i=1,n
                WRITE(*,*) "FOR #1: SVD #",i,":" ! DEBUG
                DO iii = 1, SIZE(a,1)
                   WRITE(*,*) "a(",iii,")=",a(iii,:) ! DEBUG
                ENDDO
                DO iii = 1, SIZE(w)
                   WRITE(*,*) "w(",iii,")=",w(iii) ! DEBUG
                ENDDO
                DO iii = 1, SIZE(v,1)
                   WRITE(*,*) "v(",iii,")=",v(iii,:) ! DEBUG
                ENDDO
		l=i+1
		rv1(i)=scale*g
		g=0.0
		scale=0.0
		IF (i <= m) THEN
			scale=SUM(ABS(a(i:m,i)))
			IF (scale /= 0.0) THEN
				a(i:m,i)=a(i:m,i)/scale
				s=DOT_PRODUCT(a(i:m,i),a(i:m,i))
				f=a(i,i)
				g=-SIGN(SQRT(s),f)
				h=f*g-s
				a(i,i)=f-g
				tempn(l:n)=MATMUL(a(i:m,i),a(i:m,l:n))/h
				a(i:m,l:n)=a(i:m,l:n)+outerprod(a(i:m,i),tempn(l:n))
				a(i:m,i)=scale*a(i:m,i)
			END IF
		END IF
		w(i)=scale*g
		g=0.0
		scale=0.0
		IF ((i <= m) .AND. (i /= n)) THEN
			scale=SUM(ABS(a(i,l:n)))
			IF (scale /= 0.0) THEN
				a(i,l:n)=a(i,l:n)/scale
				s=DOT_PRODUCT(a(i,l:n),a(i,l:n))
				f=a(i,l)
				g=-SIGN(SQRT(s),f)
				h=f*g-s
				a(i,l)=f-g
				rv1(l:n)=a(i,l:n)/h
				tempm(l:m)=MATMUL(a(l:m,l:n),a(i,l:n))
				a(l:m,l:n)=a(l:m,l:n)+outerprod(tempm(l:m),rv1(l:n))
				a(i,l:n)=scale*a(i,l:n)
			END IF
		END IF
	END DO
	anorm=MAXVAL(ABS(w)+ABS(rv1))
	DO i=n,1,-1
                WRITE(*,*) "FOR #2: SVD #",i,":" ! DEBUG
                DO iii = 1, SIZE(a,1)
                   WRITE(*,*) "a(",iii,")=",a(iii,:) ! DEBUG
                ENDDO
                DO iii = 1, SIZE(w)
                   WRITE(*,*) "w(",iii,")=",w(iii) ! DEBUG
                ENDDO
                DO iii = 1, SIZE(v,1)
                   WRITE(*,*) "v(",iii,")=",v(iii,:) ! DEBUG
                ENDDO
		IF (i < n) THEN
			IF (g /= 0.0) THEN
				v(l:n,i)=(a(i,l:n)/a(i,l))/g
				tempn(l:n)=MATMUL(a(i,l:n),v(l:n,l:n))
				v(l:n,l:n)=v(l:n,l:n)+outerprod(v(l:n,i),tempn(l:n))
			END IF
			v(i,l:n)=0.0
			v(l:n,i)=0.0
		END IF
		v(i,i)=1.0
		g=rv1(i)
		l=i
	END DO
	DO i=MIN(m,n),1,-1
                WRITE(*,*) "FOR #3: SVD #",i,":" ! DEBUG
                DO iii = 1, SIZE(a,1)
                   WRITE(*,*) "a(",iii,")=",a(iii,:) ! DEBUG
                ENDDO
                DO iii = 1, SIZE(w)
                   WRITE(*,*) "w(",iii,")=",w(iii) ! DEBUG
                ENDDO
                DO iii = 1, SIZE(v,1)
                   WRITE(*,*) "v(",iii,")=",v(iii,:) ! DEBUG
                ENDDO
		l=i+1
		g=w(i)
		a(i,l:n)=0.0
		IF (g /= 0.0) THEN
			g=1.0_sp/g
			tempn(l:n)=(MATMUL(a(l:m,i),a(l:m,l:n))/a(i,i))*g
			a(i:m,l:n)=a(i:m,l:n)+outerprod(a(i:m,i),tempn(l:n))
			a(i:m,i)=a(i:m,i)*g
		ELSE
			a(i:m,i)=0.0
		END IF
		a(i,i)=a(i,i)+1.0_sp
	END DO
	DO k=n,1,-1
                WRITE(*,*) "FOR #4: SVD #",k,":" ! DEBUG
                DO iii = 1, SIZE(a,1)
                   WRITE(*,*) "a(",iii,")=",a(iii,:) ! DEBUG
                ENDDO
                DO iii = 1, SIZE(w)
                   WRITE(*,*) "w(",iii,")=",w(iii) ! DEBUG
                ENDDO
                DO iii = 1, SIZE(v,1)
                   WRITE(*,*) "v(",iii,")=",v(iii,:) ! DEBUG
                ENDDO
		DO its=1,30
                        WRITE(*,*) "k=",k,"its=",its
			DO l=k,1,-1
				nm=l-1
                                WRITE(*,*) "(ABS(rv1(l))+anorm)=",(ABS(rv1(l))+anorm)
                                WRITE(*,*) "ABS((ABS(rv1(l))+anorm)-anorm)=",ABS((ABS(rv1(l))+anorm)-anorm)
                                WRITE(*,*) ((ABS(rv1(l))+anorm) == anorm)
				IF ((ABS(rv1(l))+anorm) == anorm) EXIT
                                WRITE(*,*) "nm=",nm,"l=",l
                                WRITE(*,*) "(ABS(w(nm))+anorm)=",(ABS(w(nm))+anorm)
                                WRITE(*,*) "ABS((ABS(w(nm))+anorm)-anorm)=",ABS((ABS(w(nm))+anorm)-anorm)
				IF ((ABS(w(nm))+anorm) == anorm) THEN
                                        WRITE(*,*) "TRUE: ABS((ABS(w(nm))+anorm)-anorm)=",ABS((ABS(w(nm))+anorm)-anorm)
					c=0.0
					s=1.0
					DO i=l,k
						f=s*rv1(i)
						rv1(i)=c*rv1(i)
						IF ((ABS(f)+anorm) == anorm) EXIT
						g=w(i)
						h=pythag(f,g)
						w(i)=h
						h=1.0_sp/h
						c= (g*h)
						s=-(f*h)
						tempm(1:m)=a(1:m,nm)
						a(1:m,nm)=a(1:m,nm)*c+a(1:m,i)*s
						a(1:m,i)=-tempm(1:m)*s+a(1:m,i)*c
					END DO
					EXIT
				END IF
			END DO
			z=w(k)
			IF (l == k) THEN
                                WRITE(*,*) "l == k; z=", z
				IF (z < 0.0) THEN
					w(k)=-z
					v(1:n,k)=-v(1:n,k)
				END IF
				EXIT
			END IF
			IF (its == 30) CALL nrerror('svdcmp_sp: no convergence in svdcmp')
			x=w(l)
			nm=k-1
			y=w(nm)
			g=rv1(nm)
			h=rv1(k)
			f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0_sp*h*y)
			g=pythag(f,1.0_sp)
			f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x
                        WRITE(*,*) "g=",g,"f=",f
			c=1.0
			s=1.0
			DO j=l,nm
				i=j+1
				g=rv1(i)
				y=w(i)
				h=s*g
				g=c*g
				z=pythag(f,h)
				rv1(j)=z
				c=f/z
				s=h/z
				f= (x*c)+(g*s)
				g=-(x*s)+(g*c)
				h=y*s
				y=y*c
				tempn(1:n)=v(1:n,j)
				v(1:n,j)=v(1:n,j)*c+v(1:n,i)*s
				v(1:n,i)=-tempn(1:n)*s+v(1:n,i)*c
				z=pythag(f,h)
				w(j)=z
				IF (z /= 0.0) THEN
					z=1.0_sp/z
					c=f*z
					s=h*z
				END IF
				f= (c*g)+(s*y)
				x=-(s*g)+(c*y)
				tempm(1:m)=a(1:m,j)
				a(1:m,j)=a(1:m,j)*c+a(1:m,i)*s
				a(1:m,i)=-tempm(1:m)*s+a(1:m,i)*c
			END DO
			rv1(l)=0.0
			rv1(k)=f
			w(k)=x
		END DO
	END DO
	END SUBROUTINE svdcmp_sp

	SUBROUTINE svbksb_sp(u,w,v,b,x)
	USE nrtype; USE nrutil, ONLY : assert_eq
	REAL(SP), DIMENSION(:,:), INTENT(IN) :: u,v
	REAL(SP), DIMENSION(:), INTENT(IN) :: w,b
	REAL(SP), DIMENSION(:), INTENT(OUT) :: x
	INTEGER(I4B) :: mdum,ndum
	REAL(SP), DIMENSION(SIZE(x)) :: tmp
	mdum=assert_eq(SIZE(u,1),SIZE(b),'svbksb_sp: mdum')
	ndum=assert_eq((/SIZE(u,2),SIZE(v,1),SIZE(v,2),SIZE(w),SIZE(x)/),&
		'svbksb_sp: ndum')
	WHERE (w /= 0.0)
		tmp=MATMUL(b,u)/w
	ELSEWHERE
		tmp=0.0
	END WHERE
	x=MATMUL(v,tmp)
	END SUBROUTINE svbksb_sp

	FUNCTION pythag(a,b)
	USE nrtype
	IMPLICIT NONE
	REAL(SP), INTENT(IN) :: a,b
	REAL(SP) :: pythag
	REAL(SP) :: absa,absb
	absa=ABS(a)
	absb=ABS(b)
	IF (absa > absb) THEN
		pythag=absa*SQRT(1.0_sp+(absb/absa)**2)
	ELSE
		IF (absb == 0.0) THEN
			pythag=0.0
		ELSE
			pythag=absb*SQRT(1.0_sp+(absa/absb)**2)
		END IF
	END IF
	END FUNCTION pythag



  
     
    




