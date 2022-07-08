      program kgrid
      use kgrid_var
      implicit none
      integer*4 :: i,j

      call system('clear')
      write(*,*)  '============== KGRID =============='
      write(*,*)  '===== Grid Generation START ! ====='
      write(*,*)  '==================================='
      write(*,*)

c---------------------------------------------- Initial
      call InitMat

c----------------------------------------------- CalMat
      do j=2,NJ
      call CalMat(j)

c------------------------------------------------- TDMA
      call TDMA(NI,AA,BB,CC,DD,RR,Fk)

c     call Smoothing(NI,NJ,j,RR)

      Z(:,j,1) = RR(:,1,1)
      Z(:,j,2) = RR(:,2,1)

      end do

      open(5,file='HypGrid.dat')
      write(5,*) 'variables = "X","Y","F","Fk"'
      write(5,*) 'zone i=',NI,' j=',NJ
      do j=1,NJ+1
         do i=1,NI
            write(5,100) Z(i,j,1),Z(i,j,2),F(i),Fk(i)
         end do
      end do
      close(5)

      deallocate(AA,BB,CC,DD,RR)
      deallocate(Z,X,Y,F,Fk)
      write(*,*)
      write(*,*)  '============== KGRID =============='
      write(*,*)  '===== Grid Generation END !!! ====='
      write(*,*)  '==================================='
100   format(1x,4(e20.12,1x))
      stop
      end program
c------------------------------------------------------
c------------------------------------------------------
      subroutine Smoothing(NI,NJ,j,R)
      implicit none
      integer*4 :: NI,NJ
      integer*4 :: i,j,k
      real(kind=8),dimension(NI,2,1) :: R,R0
      real(kind=8) :: alpha
      real(kind=8) :: lm,lp

      alpha = 0.5d0
      alpha = min(alpha,0.5d0*(alpha+alpha*(j-2.0d0)/NJ))
c     alpha = min(alpha,alpha*(j-2.0d0)/NJ)
c     alpha = 0.5d0

      do k=1,3
      R0    = R
      do i = 2,NI-1
         lp = (R0(i+1,1,1)-R0(i,1,1))**2.0d0
         lp = (R0(i+1,2,1)-R0(i,2,1))**2.0d0 + lp
         lp = dsqrt(lp)

         lm = (R0(i,1,1)-R0(i-1,1,1))**2.0d0
         lm = (R0(i,2,1)-R0(i-1,2,1))**2.0d0 + lm
         lm = dsqrt(lm)

         R(i,1,1) = (1.0d0-alpha)*R0(i,1,1) + 
     +        alpha*(lm*R0(i+1,1,1)+lp*R0(i-1,1,1))/(lp+lm)
c    +        alpha*(0.5*R0(i+1,1,1)+0.5*R0(i-1,1,1))
         R(i,2,1) = (1.0d0-alpha)*R0(i,2,1) + 
     +        alpha*(lm*R0(i+1,2,1)+lp*R0(i-1,2,1))/(lp+lm)
c    +        alpha*(0.5*R0(i+1,2,1)+0.5*R0(i-1,2,1))
      end do
      end do


c     do j=1,2
c     R0    = R
c     do i = 50,50
c        lp = (R0(i+1,1,1)-R0(i,1,1))**2.0d0
c        lp = (R0(i+1,2,1)-R0(i,2,1))**2.0d0 + lp
c        lp = dsqrt(lp)
c
c        lm = (R0(i,1,1)-R0(i-1,1,1))**2.0d0
c        lm = (R0(i,2,1)-R0(i-1,2,1))**2.0d0 + lm
c        lm = dsqrt(lm)
c
c        R(i,1,1) = (1.0d0-alpha)*R0(i,1,1) + 
c    +        alpha*(lm*R0(i+1,1,1)+lp*R0(i-1,1,1))/(lp+lm)
c        R(i,2,1) = (1.0d0-alpha)*R0(i,2,1) + 
c    +        alpha*(lm*R0(i+1,2,1)+lp*R0(i-1,2,1))/(lp+lm)
c     end do
c     end do






      return
      end
c------------------------------------------------------
