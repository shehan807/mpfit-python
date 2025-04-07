PROGRAM test_svd
  USE nrtype
  USE nrutil, ONLY : assert_eq
  IMPLICIT NONE
  
  ! Declare variables
  INTEGER, PARAMETER :: m = 3, n = 3
  REAL(SP), DIMENSION(m,n) :: a, a_test
  REAL(SP), DIMENSION(n) :: w
  REAL(SP), DIMENSION(n,n) :: v
  REAL(SP), DIMENSION(m,n) :: reconstructed
  REAL(SP) :: max_diff
  INTEGER :: i, j, k
  
  ! Initialize test matrix (same as Python test)
  a(1,1) = 1.0_sp
  a(1,2) = 2.0_sp
  a(1,3) = 3.0_sp
  a(2,1) = 4.0_sp
  a(2,2) = 5.0_sp
  a(2,3) = 6.0_sp
  a(3,1) = 7.0_sp
  a(3,2) = 8.0_sp
  a(3,3) = 9.0_sp
  
  ! Make a copy for the test
  a_test = a
  
  ! Print the test matrix
  WRITE(*,*) "Running SVD decomposition on test matrix:"
  DO i = 1, m
     WRITE(*,*) (a_test(i,j), j=1,n)
  ENDDO
  WRITE(*,*)
  
  ! Run the SVD decomposition
  CALL svdcmp_sp(a_test, w, v)
  
  ! Print the results
  WRITE(*,*) "Results of SVD decomposition:"
  WRITE(*,*) "U matrix (stored in a):"
  DO i = 1, m
     WRITE(*,*) (a_test(i,j), j=1,n)
  ENDDO
  WRITE(*,*)
  
  WRITE(*,*) "Singular values (w):"
  WRITE(*,*) (w(j), j=1,n)
  WRITE(*,*)
  
  WRITE(*,*) "V matrix:"
  DO i = 1, n
     WRITE(*,*) (v(i,j), j=1,n)
  ENDDO
  WRITE(*,*)
  
  ! Verify the decomposition
  ! U * diag(w) * V^T should be close to the original matrix
  reconstructed = 0.0_sp
  DO i = 1, m
     DO j = 1, n
        DO k = 1, n
           reconstructed(i,j) = reconstructed(i,j) + a_test(i,k) * w(k) * v(j,k)
        ENDDO
     ENDDO
  ENDDO
  
  WRITE(*,*) "Reconstructed matrix (U * diag(w) * V^T):"
  DO i = 1, m
     WRITE(*,*) (reconstructed(i,j), j=1,n)
  ENDDO
  WRITE(*,*)
  
  WRITE(*,*) "Difference from original matrix:"
  DO i = 1, m
     WRITE(*,*) (a(i,j) - reconstructed(i,j), j=1,n)
  ENDDO
  WRITE(*,*)
  
  ! Calculate maximum absolute difference
  max_diff = 0.0_sp
  DO i = 1, m
     DO j = 1, n
        max_diff = MAX(max_diff, ABS(a(i,j) - reconstructed(i,j)))
     ENDDO
  ENDDO
  
  WRITE(*,*) "Maximum absolute difference:", max_diff
  
END PROGRAM test_svd 