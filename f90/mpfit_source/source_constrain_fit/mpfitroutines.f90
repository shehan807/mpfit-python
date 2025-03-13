module mpfitroutines

contains

!*********************************
! this subroutine determines the number of multipole sites
! that are read in as input.  This should be the number of 
! atoms in the molecule
!*********************************
  SUBROUTINE numbersites(inpfile,n)
    CHARACTER(*),INTENT(in)::inpfile
    INTEGER,INTENT(out)::n
    INTEGER::count,inputstatus,maxl,i
    real*8::x,y,z
    CHARACTER(50)::line
    character(3)::type
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


!*********************************
! this subroutine reads the input multipole moments
! from the input file
!*********************************
  SUBROUTINE getmultmoments(inpfile,n,lmax,mm,ms,atomtype)
    CHARACTER(*),INTENT(in)::inpfile
    INTEGER,INTENT(in)::n
    integer,dimension(:),intent(out)::lmax
    REAL*8,DIMENSION(:,0:,0:,0:),INTENT(out)::mm
    REAL*8,DIMENSION(:,:),INTENT(out)::ms
    character(3),dimension(:),intent(out)::atomtype
    INTEGER::i,l,num
    REAL*8::x,y,z,q0


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

  END SUBROUTINE getmultmoments

  SUBROUTINE gencharges(ms,qs)
    REAL*8,DIMENSION(:,:),INTENT(in)::ms
    REAL*8,DIMENSION(:,:),INTENT(out)::qs
    INTEGER::i,j,nmult,count

    nmult=SIZE(ms(:,1))
    nmid=SIZE(qs(:,1))-SIZE(ms(:,1))

    DO i=1,nmult
       qs(i,1)=ms(i,1)
       qs(i,2)=ms(i,2)
       qs(i,3)=ms(i,3)
    ENDDO


  END SUBROUTINE gencharges



  !************************************
  ! this subroutine determines the vdws radius of each atom,
  ! which is used to determine the surface for which to fit the
  ! electrostatic potential of each multipole distribution
  ! currently use one value for heavy atoms, and a special value
  ! for hydrogen atoms
  ! See:: JCC,12,(1991), 913-917
  !************************************
  subroutine genvdw(rvdw,atomtype)
    real*8,dimension(:),intent(out)::rvdw
    character(3),dimension(:),intent(in)::atomtype
    integer::i,count
!!!!!!!!!!default
    rvdw(:)=3.0
!!!!!!!!!!!!!!!!!!!!
    do i=1,size(rvdw)
       count=index(atomtype(i),'H')
       if (count .ne. 0) then
          rvdw(i)=2.27
       endif
    enddo

  end subroutine genvdw


  subroutine sitespecificdata
    use variables
    integer::i,j,multsites,midsites,count
    real*8::rqm

    multsites=size(xyzmult(:,1));midsites=size(xyzcharge(:,1))-multsites
    i=site
    countatom=0
    countmid=0
    DO j=1,multsites
       rqm=((xyzmult(i,1)-xyzcharge(j,1))**2+(xyzmult(i,2)-xyzcharge(j,2))**2+&
            & (xyzmult(i,3)-xyzcharge(j,3))**2)**.5
!!$        IF (rqm < rvdw(i)+r1) THEN
       IF (rqm < rvdw(i)) THEN
          quse(site,j)=1
          countatom=countatom+1
       ELSE
          quse(site,j)=0
       ENDIF
    ENDDO

  end subroutine sitespecificdata


  SUBROUTINE numindatommid(atomtype,atms)
    CHARACTER(3),DIMENSION(:),INTENT(in)::atomtype
    INTEGER,INTENT(out)::atms
    CHARACTER(3)::value
    INTEGER::count1,count2,atoms,i,j
    CHARACTER(4),DIMENSION(:),ALLOCATABLE::typemon1

    atoms=SIZE(atomtype)


    ALLOCATE(typemon1(atoms))

    DO i=1,atoms
       WRITE(typemon1(i),'(A3)') atomtype(i)
    ENDDO
    count1=0
    count2=0
    DO i=1,atoms
       IF(i < atoms) THEN
          DO j=i+1,atoms
             IF(typemon1(i) .EQ. typemon1(j)) THEN
                count1=count1+1
                WRITE(value,'(I3)') j
                typemon1(j)='z'//value
             ENDIF
          ENDDO
       ENDIF
    ENDDO

    atms=atoms-count1

  END SUBROUTINE numindatommid


  subroutine collapseatommid(atomtypemon1,atmtypemon1)
    character(3),dimension(:),intent(out)::atmtypemon1
    CHARACTER(3),DIMENSION(:),INTENT(in)::atomtypemon1
    INTEGER::count,countt,atoms1,atms1,i,j,k

    atoms1=SIZE(atomtypemon1);
    atms1=SIZE(atmtypemon1);


    count=0
    countt=0
    atmtypemon1(1)=atomtypemon1(1)
    if(atms1 > 1) then
       DO i=2,atms1
          count=countt
          DO j=i+count,atoms1
             DO k=1,j-1
                IF(atomtypemon1(k).EQ.atomtypemon1(j)) THEN
                   countt=countt+1
                   go to 200 
                ENDIF
             ENDDO
             atmtypemon1(i)=atomtypemon1(j)
             go to 201
200          CONTINUE
          ENDDO
201       CONTINUE
       ENDDO
    endif


  END SUBROUTINE collapseatommid




end module mpfitroutines


!*****************************************************
! this is the fitting function, which is minimized to obtain the
! final parameters
!*****************************************************
function kaisq(p0)
  use variables
  real*8::kaisq
  real*8,dimension(:),intent(in)::p0
  REAL*8,DIMENSION(:),ALLOCATABLE::W
  real*8,dimension(:),allocatable::q0
  real*8,dimension(3)::xyz
  real*8,dimension(:,:),allocatable::xyzqatom,xyzqmid
  integer::i,j,l,m,s,natom,nmid,multsites
  real*8::sum,sum1,rmax,rminn,sumkai,sumcon,sumchg
  interface
     FUNCTION RSH(l,m,cs,xyz)
       REAL*8::RSH
       INTEGER,INTENT(in)::l,m,cs
       REAL*8,dimension(3),INTENT(in)::xyz
     end function RSH
     subroutine expandcharge(p0)
       real*8,dimension(:),intent(in)::p0
     end subroutine expandcharge
  end interface

  call expandcharge(p0)
  multsites=size(xyzmult(:,1));natom=multsites
  nmid=size(xyzcharge(:,1))-natom
  ALLOCATE(W(0:maxl),q0(natom+nmid),xyzqatom(natom,3),xyzqmid(nmid,3))
  do i=1,natom
     xyzqatom(i,:)=xyzcharge(i,:)
  enddo
  if(nmid>0) then
     do i=1,nmid
        xyzqmid(i,:)=xyzcharge(natom+i,:)
     enddo
  endif
  sumkai=0.0
  do s=1,multsites

     q0(:)=allcharge(s,:)
     site=s
     rmax=rvdw(site)+r2
     rminn=rvdw(site)+r1

     do i=0,lmax(s)
        W(i)=(1.0/(1.0-2.0*i))*(rmax**(1-2*i)-rminn**(1-2*i))
     ENDDO

     sum=0.0
     do l=0,lmax(s)

        IF(l .EQ. 0) THEN
           sum1=0.0
           do j=1,natom
              xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
              sum1=sum1+q0(j)*RSH(0,0,0,xyz)
           enddo
           if(nmid>0) then
              do j=1,nmid
                 xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                 sum1=sum1+(q0(natom+j))*RSH(0,0,0,xyz)
              enddo
           endif
           sum=(4.0*PI/(2.0*l+1.0))*W(0)*(multipoles(site,l,0,0)-sum1)**2
!!$        write(*,*) l,sum
        ELSE
           DO m=0,l
              IF(m .EQ. 0) THEN
                 sum1=0.0
                 do j=1,natom
                    xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
                    sum1=sum1+q0(j)*RSH(l,0,0,xyz)
                 enddo
                 if(nmid>0) then
                    do j=1,nmid
                       xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                       sum1=sum1+(q0(natom+j))*RSH(l,0,0,xyz)
                    enddo
                 endif
                 sum=sum+(4.0*PI/(2.0*l+1.0))*W(l)*(multipoles(site,l,0,0)-sum1)**2
!!$              write(*,*) l,sum
              ELSE
                 sum1=0.0
                 do j=1,natom
                    xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
                    sum1=sum1+q0(j)*RSH(l,m,0,xyz)
                 enddo
                 if(nmid>0) then
                    do j=1,nmid
                       xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                       sum1=sum1+(q0(natom+j))*RSH(l,m,0,xyz)
                    enddo
                 endif
                 sum=sum+(4.0*PI/(2.0*l+1.0))*W(l)*(multipoles(site,l,m,0)-sum1)**2
!!$              write(*,*) l,sum

                 sum1=0.0
                 do j=1,natom
                    xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
                    sum1=sum1+q0(j)*RSH(l,m,1,xyz)
                 enddo
                 if(nmid>0) then
                    do j=1,nmid
                       xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                       sum1=sum1+(q0(natom+j))*RSH(l,m,1,xyz)
                    enddo
                 endif
                 sum=sum+(4.0*PI/(2.0*l+1.0))*W(l)*(multipoles(site,l,m,1)-sum1)**2
!!$              write(*,*) l,sum
              ENDIF
           ENDDO
        ENDIF
     ENDDO

     sumkai=sumkai+sum
  enddo

!!!!!!!!!!!!!!!!now add constraints
  sumchg=0.0
  do i=1,size(qstore)
     sumchg=sumchg+qstore(i)
  enddo
  sumcon=conchg*(sumchg-molecule_charge)**2
  if(nmid>0) then
     do i=1,nmid
        if(qstore(natom+i)>0.0) then
           sumcon=sumcon+negmidc*exp(negmide*qstore(natom+i))
        endif
     enddo
  endif


  kaisq=sumkai+sumcon


end function kaisq



!*****************************************************
! this is the derivative of the fitting function, which is used in
! a conjugate gradient function minimization routine
!*****************************************************
function dkaisq(p0)
  use variables
  real*8,dimension(:),intent(in)::p0
  real*8,dimension(size(p0))::dkaisq
  REAL*8,DIMENSION(:),ALLOCATABLE::W
  real*8,dimension(:),allocatable::q0,dparam,dparam1
  real*8,dimension(:,:),allocatable::xyzqatom,xyzqmid
  integer::i,j,l,s,m,natom,nmid,multsites,npts
  real*8::sum,sum1,rmax,rminn,sumcon,sumchg
  real*8,dimension(3)::xyz
  interface
     FUNCTION RSH(l,m,cs,xyz)
       REAL*8::RSH
       INTEGER,INTENT(in)::l,m,cs
       REAL*8,dimension(3),INTENT(in)::xyz
     end function RSH
     subroutine expandcharge(p0)
       use variables
       real*8,dimension(:),intent(in)::p0
     end subroutine expandcharge
     subroutine createdkaisq(dkaisq,dparam,atomtype,quse)
       real*8,dimension(:),intent(out)::dkaisq
       real*8,dimension(:),intent(in)::dparam
       character(3),dimension(:),intent(in)::atomtype
       integer,dimension(:,:),intent(in)::quse
     end subroutine createdkaisq
  end interface

  call expandcharge(p0)
  multsites=size(xyzmult(:,1));natom=multsites
  npts=size(xyzcharge(:,1));nmid=npts-natom
  ALLOCATE(W(0:maxl),q0(natom+nmid),xyzqatom(natom,3),xyzqmid(nmid,3))
  do i=1,natom
     xyzqatom(i,:)=xyzcharge(i,:)
  enddo
  if(nmid>0) then
     do i=1,nmid
        xyzqmid(i,:)=xyzcharge(natom+i,:)
     enddo
  endif

  ALLOCATE(dparam(multsites*(natom+nmid)),dparam1(multsites*(natom+nmid)))
  dparam=0.0
  do s=1,multsites
     q0(:)=allcharge(s,:)
     site=s
     rmax=rvdw(site)+r2
     rminn=rvdw(site)+r1
     do i=0,lmax(s)
        W(i)=(1.0/(1.0-2.0*i))*(rmax**(1-2*i)-rminn**(1-2*i))
     ENDDO
     do l=0,lmax(s)
        IF(l .EQ. 0) THEN
           sum1=0.0
           do j=1,natom
              xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
              sum1=sum1+q0(j)*RSH(0,0,0,xyz)
           enddo
           if(nmid>0) then
              do j=1,nmid
                 xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                 sum1=sum1+q0(natom+j)*RSH(0,0,0,xyz)
              enddo
           endif
           sum=2.*(4.0*PI/(2.0*l+1.0))*W(0)*(multipoles(site,l,0,0)-sum1)
           do j=1,natom
              xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
              dparam((s-1)*npts+j)=dparam((s-1)*npts+j)-sum*RSH(0,0,0,xyz)
           enddo
           if (nmid>0) then
              do j=1,nmid
                 xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                 dparam((s-1)*npts+natom+j)=dparam((s-1)*npts+natom+j)-sum*RSH(0,0,0,xyz)
              enddo
           endif
        ELSE
           DO m=0,l
              IF(m .EQ. 0) THEN
                 sum1=0.0
                 do j=1,natom
                    xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
                    sum1=sum1+q0(j)*RSH(l,0,0,xyz)
                 enddo
                 if(nmid>0) then
                    do j=1,nmid
                       xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                       sum1=sum1+q0(natom+j)*RSH(l,0,0,xyz)
                    enddo
                 endif
                 sum=2.*(4.0*PI/(2.0*l+1.0))*W(l)*(multipoles(site,l,0,0)-sum1)
                 do j=1,natom
                    xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
                    dparam((s-1)*npts+j)=dparam((s-1)*npts+j)-sum*RSH(l,0,0,xyz)
                 enddo
                 if(nmid>0) then
                    do j=1,nmid
                       xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                       dparam((s-1)*npts+natom+j)=dparam((s-1)*npts+natom+j)-sum*RSH(l,0,0,xyz)
                    enddo
                 endif
              ELSE
                 sum1=0.0
                 do j=1,natom
                    xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
                    sum1=sum1+q0(j)*RSH(l,m,0,xyz)
                 enddo
                 if(nmid>0) then
                    do j=1,nmid
                       xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                       sum1=sum1+q0(natom+j)*RSH(l,m,0,xyz)
                    enddo
                 endif
                 sum=2.*(4.0*PI/(2.0*l+1.0))*W(l)*(multipoles(site,l,m,0)-sum1)
                 do j=1,natom
                    xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
                    dparam((s-1)*npts+j)=dparam((s-1)*npts+j)-sum*RSH(l,m,0,xyz)
                 enddo
                 if(nmid>0) then
                    do j=1,nmid
                       xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                       dparam((s-1)*npts+natom+j)=dparam((s-1)*npts+natom+j)-sum*RSH(l,m,0,xyz)
                    enddo
                 endif
                 sum1=0.0
                 do j=1,natom
                    xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
                    sum1=sum1+q0(j)*RSH(l,m,1,xyz)
                 enddo
                 if (nmid>0) then
                    do j=1,nmid
                       xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                       sum1=sum1+q0(natom+j)*RSH(l,m,1,xyz)
                    enddo
                 endif
                 sum=2.*(4.0*PI/(2.0*l+1.0))*W(l)*(multipoles(site,l,m,1)-sum1)
                 do j=1,natom
                    xyz(:)=xyzqatom(j,:)-xyzmult(site,:)
                    dparam((s-1)*npts+j)=dparam((s-1)*npts+j)-sum*RSH(l,m,1,xyz)
                 enddo
                 if(nmid>0) then
                    do j=1,nmid
                       xyz(:)=xyzqmid(j,:)-xyzmult(site,:)
                       dparam((s-1)*npts+natom+j)=dparam((s-1)*npts+natom+j)-sum*RSH(l,m,1,xyz) 
                    enddo
                 endif
              ENDIF
           ENDDO
        ENDIF
     ENDDO
  enddo

!!!!!!!!!!!!!!!!!!!!!!!now change ordering of dparam and store in dparam1
  do i=1,multsites
     do j=1,npts
        dparam1((j-1)*multsites+i)=dparam((i-1)*npts+j)
     enddo
  enddo

!!!!!!!!!!!!!!!!!!!!!!!!now add constraints
  sumchg=0.
  do i=1,size(qstore)
     sumchg=sumchg+qstore(i)
  enddo
  dparam1=dparam1+conchg*2.*(sumchg-molecule_charge)
  if(nmid>0) then
     do i=1,nmid
        if(qstore(natom+i)>0.0) then
           do j=1,multsites
              dparam1((natom+i-1)*multsites+j)=dparam1((natom+i-1)*multsites+j)+negmide*negmidc*exp(negmide*qstore(natom+i))
           enddo
        endif
     enddo
  endif
  call createdkaisq(dkaisq,dparam1,atomtype,quse)

end function dkaisq


!*********************************
!   this function evaluates regular spherical harmonics
!   at point x,y,z where l,m,cs determine the rank
!   cs=0 means cosine, cs=1 means sine
!  
!*********************************
FUNCTION RSH(l,m,cs,xyz)
  REAL*8::RSH
  INTEGER,INTENT(in)::l,m,cs
  REAL*8,dimension(3),INTENT(in)::xyz
  REAL*8::rsq,x,y,z
  REAL*8,DIMENSION(:,:,:),ALLOCATABLE::rsharray

  ALLOCATE(rsharray(0:4,0:4,0:1))
  x=xyz(1);y=xyz(2);z=xyz(3)
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



!********************************
! this subroutine adds up the total charge on each atom,
! which has contributions from each multipole distribution
! that is separately fit
!********************************
subroutine expandcharge(p0)
  use variables
  real*8,dimension(:),intent(in)::p0
  integer::i,j,k,atoms,nmid,count,multsites,count1,count2,twin
  real*8::sum

  multsites=size(xyzmult(:,1))
  atoms=size(atomtype)
  nmid=0
  allcharge=0.0


  count=1
  do i=1,atoms
     count1=0
     sum=0.0
     if(i.eq.1) then
        do j=1,multsites
           if(quse(j,i).eq.1) then
              allcharge(j,i)=p0(count)
              sum=sum+p0(count)
              count=count+1
           endif
        enddo
        qstore(i)=sum
     else
        twin=0
        do k=1,i-1
           if (atomtype(i).eq.atomtype(k)) twin=k
        enddo
        if(twin.ne.0) then
           do j=1,multsites
              if(quse(j,i).eq.1) then
                 count1=count1+1
              endif
           enddo
           count2=1
           do j=1,multsites
              if((quse(j,i).eq.1).and.(count2<count1)) then
                 allcharge(j,i)=p0(count)
                 sum=sum+p0(count)
                 count=count+1
                 count2=count2+1
              elseif((quse(j,i).eq.1).and.(count2.eq.count1))then
                 allcharge(j,i)=qstore(twin)-sum
                 qstore(i)=qstore(twin)
              endif
           enddo
        else
           do j=1,multsites
              if(quse(j,i).eq.1) then
                 allcharge(j,i)=p0(count)
                 sum=sum+p0(count)
                 count=count+1
              endif
           enddo
           qstore(i)=sum
        endif
     endif
  enddo

  if (nmid>0) then
     do i=1,nmid
        count1=0
        sum=0.0
        if(i.eq.1) then
           do j=1,multsites
              if(quse(j,atoms+i).eq.1) then
                 allcharge(j,atoms+i)=p0(count)
                 sum=sum+p0(count)
                 count=count+1
              endif
           enddo
           qstore(atoms+i)=sum
        else
              do j=1,multsites
                 if(quse(j,atoms+i).eq.1) then
                    allcharge(j,atoms+i)=p0(count)
                    sum=sum+p0(count)
                    count=count+1
                 endif
              enddo
              qstore(atoms+i)=sum
        endif
     enddo
  endif
end subroutine expandcharge


subroutine createdkaisq(dkaisq,dparam1,atomtype,quse)
  real*8,dimension(:),intent(out)::dkaisq
  real*8,dimension(:),intent(inout)::dparam1
  character(3),dimension(:),intent(in)::atomtype
  integer,dimension(:,:),intent(in)::quse
  integer,dimension(:),allocatable::sameatom,samemid,nparam
  integer::i,j,k,twin,count,count1,count2,multsites,atoms,nmid
  integer,dimension(:),allocatable::fill
  real*8::sum

  atoms=size(atomtype)
  nmid=0
  multsites=atoms

!!$do i=1,2
!!$   write(*,*) i, (quse(j,i),j=1,multsites)
!!$   write(*,*) i, (dparam1((i-1)*multsites+j),j=1,multsites)
!!$enddo
!!!!!!!!!!!!!!!!first combine derivatives for atoms of same type
  do i=1,atoms
     twin=0
     if(i.ne.1) then
        do k=1,i-1
           if (atomtype(i).eq.atomtype(k)) then
              twin=k
              goto 100
           endif
        enddo
100     continue
        if(twin.ne.0) then
           count1=0
           do j=1,multsites
              if(quse(j,i).eq.1) then
                 count1=count1+1
              endif
           enddo
           count2=1 
           do j=1,multsites
              if((quse(j,i).eq.1).and.(count2.lt.count1)) then
                 count2=count2+1
              elseif((quse(j,i).eq.1).and.(count2.eq.count1)) then
                 do k=1,multsites
                    dparam1((twin-1)*multsites+k)=dparam1((twin-1)*multsites+k)+dparam1((i-1)*multsites+j)
                 enddo
                 do k=1,j-1
                    dparam1((i-1)*multsites+k)=dparam1((i-1)*multsites+k)-dparam1((i-1)*multsites+j)
                 enddo
              endif
           enddo
        endif
     endif
  enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!now order derivatives into dkaisq
  count=1
  do i=1,atoms
     if(i.eq.1) then
        do j=1,multsites
           if(quse(j,i).eq.1) then
              dkaisq(count)=dparam1((i-1)*multsites+j)
              count=count+1
           endif
        enddo
     else
        twin=0
        do k=1,i-1
           if (atomtype(i).eq.atomtype(k)) then
              twin=k
           endif
        enddo
        if(twin.ne.0) then
           count1=0
           do j=1,multsites
              if(quse(j,i).eq.1) then
                 count1=count1+1
              endif
           enddo
           count2=1 
           do j=1,multsites
              if((quse(j,i).eq.1).and.(count2.lt.count1)) then
                 dkaisq(count)=dparam1((i-1)*multsites+j)
                 count2=count2+1
                 count=count+1
              endif
           enddo
        else
           do j=1,multsites
              if(quse(j,i).eq.1) then
                 dkaisq(count)=dparam1((i-1)*multsites+j)
                 count=count+1
              endif
           enddo
        endif
     endif
  enddo


end subroutine createdkaisq
