PROGRAM mpfit

  USE interfac
  REAL*8,PARAMETER:: r1=6.78,r2=12.45,small=1.0D-4 !r1=5.26,r2=8.67
  Integer,parameter::maxl=4  !!!!!!maxl is the highest l value of all the sites
  CHARACTER(50)::inpfile,chargefile
  INTEGER::multsites,chargesites,count,i,j,k,qsites,printt,hyd
  REAL*8::rqm
  INTEGER,DIMENSION(:),ALLOCATABLE::quse,lmax
  INTEGER,DIMENSION(:,:),ALLOCATABLE::midbond
  REAL*8,DIMENSION(:),ALLOCATABLE::q,b,qstore,w,btst,rvdw
  REAL*8,DIMENSION(:,:),ALLOCATABLE::xyzmult,xyzcharge,xyzq
  REAL*8,DIMENSION(:,:),ALLOCATABLE::A,Astore,v
  REAL*8,DIMENSION(:,:,:,:),ALLOCATABLE::multipoles
  character(2),dimension(:),allocatable::atomtype
  CALL getarg(1,inpfile)

  CALL numbersites(inpfile,multsites)

  ALLOCATE(multipoles(multsites,0:maxl,0:maxl,0:1),xyzmult(multsites,3))
  allocate(midbond(multsites,multsites),lmax(multsites),rvdw(multsites),atomtype(multsites))

!!!!!!!!!!!!!!!!!!!!!!!!!!!!put 1's in midbond array where you want
  midbond(:,:)=0
  !!
  !!  put 1's in upper triangle of matrix
  !!  so for three atoms, midbond between 1,2 and 2,3
  !!
  !!         (0  1  0
  !!          0  0  1
  !!          0  0  0)
  !!

!!!!!  allocate size of charge array
  count=0
  DO i=1,multsites
     DO j= i+1,multsites
        IF (midbond(i,j).EQ.1) THEN
           count=count+1
        ENDIF
     ENDDO
  ENDDO

  chargesites=multsites+count
  ALLOCATE(xyzcharge(chargesites,3),qstore(chargesites),quse(chargesites))
  qstore(:)=0.0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


  CALL getmultmoments(inpfile,multsites,lmax,multipoles,xyzmult,atomtype)

  CALL gencharges(xyzmult,xyzcharge,midbond)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!create rvdw, which determines radius encompassing charges
!!!for each multipole site.  For instance, if there is only a monopole
!!!on hydrogen, make rvdw small so that the monopole is put on hydrogen
!!!!!!
!!!!!!
!!!!!!!!!!default
      rvdw(:)=3.0
!!!!!!!!!!!!!!!!!!!!!!change for hydrogen
do i=1,multsites
 hyd=index(atomtype(i),"H")
 if(hyd.eq.0) then
rvdw(i)=3.0
else
!!$rvdw(i)=1.0
endif
enddo



!!!!!!!!!!!!!!!!!!!fit charges for each multipole site, and then add them
!!!!!!!!!!!!!!!!!! to the total charge aray qstore

  DO i=1,multsites

!!!!!!!!determine which charge positions are close enough to fit given multsite
     count=0
     DO j=1,chargesites
        rqm=((xyzmult(i,1)-xyzcharge(j,1))**2+(xyzmult(i,2)-xyzcharge(j,2))**2+(xyzmult(i,3)-xyzcharge(j,3))**2)**.5
        IF (rqm < rvdw(i)) THEN
           quse(j)=1
           count=count+1
        ELSE
           quse(j)=0
        ENDIF
     ENDDO
     qsites=count

!!!!!!!!!!!!!!!!!!allocate A,q,b,etc to correct dimensionality for given site
     IF(ALLOCATED(A)) THEN
        DEALLOCATE(A,Astore,v,w,q,b,xyzq,btst)
     ENDIF
     ALLOCATE(A(qsites,qsites),Astore(qsites,qsites),v(qsites,qsites),w(qsites),q(qsites),b(qsites),xyzq(qsites,3),btst(qsites))

!!!!!!!!!!!!!generate xyzq array from xyzcharge array picking out relevant charges in order from lowest label site
     count=1
     DO j=1,chargesites
        IF (quse(j).EQ.1) THEN
           xyzq(count,1)=xyzcharge(j,1)
           xyzq(count,2)=xyzcharge(j,2)
           xyzq(count,3)=xyzcharge(j,3)
           count=count+1
        ENDIF
     ENDDO


     CALL Amat(i,xyzmult,xyzq,r1,r2,lmax(i),A)
     CALL bvec(i,xyzmult,xyzq,r1,r2,lmax(i),multipoles,b)

     Astore(:,:)=A(:,:)

     CALL svdcmp_sp(A,w,v)

!!!!!!!!!!!in the singular value decomposition, set any small values equal to zero
     DO j=1, SIZE(w(:))
        IF(w(j) < small) THEN
           w(j)=0.0
        ENDIF
     ENDDO
     CALL svbksb_sp(A,w,v,b,q)

!!!!!!!!!!!!!!!!!test svd
     btst=MATMUL(Astore,q)

!!$write(*,*) i
!!$  do j=1,qsites
!!$    write(*,*) q(j)
!!$ write(*,*) btst(j),b(j)
!!$  enddo

!!!!!!!!!!!!!!!!!!!!!for the fit charges q, add them to the total array qstore

     count=1
     DO j=1, chargesites
        IF(quse(j).EQ.1) THEN
           qstore(j)=qstore(j)+q(count)
           count=count+1
        ENDIF
     ENDDO

  ENDDO

  DO j=1,multsites
      WRITE(*,'(A,7x,F8.5)') atomtype(j),qstore(j)
  ENDDO


 END PROGRAM mpfit

       
      


