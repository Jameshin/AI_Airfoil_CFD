c------------------------------------------------------
      subroutine CalMat(j)
      use kgrid_var
      implicit none
      integer*4 :: i,j,IM1
      real(kind=8),dimension(NI) :: xxi,xet,yxi,yet
      real(kind=8),dimension(2,2):: MatA,MatB,MatI,InvB,MatC
      real(kind=8),dimension(2,1):: MatH,MatD,MatR
      real(kind=8) :: dx1,dx2,dy1,dy2,delc,dels,dis,dx,dy
      real(kind=8) :: a1,a2,b1,b2,alpha
      real(kind=8) :: area_mean,factor,factor_blend

      AA=0.0d0;BB=0.0d0;CC=0.0d0;DD=0.0d0;RR=0.0d0
      X =0.0d0;Y =0.0d0

      factor_blend = 0.4d0

      X(:) = Z(:,j-1,1)
      Y(:) = Z(:,j-1,2)
      RR(:,1,1) = X(:)
      RR(:,2,1) = Y(:)
      dels = dis_first * (1.0d0 + dis_ratio)**(j-2.0d0) !marching distance 

      IM1 = NI - 1
c---- xxi,yxi,xet,yet
      do i=2,IM1
         xxi(i) = (X(i+1)-X(i-1))*0.5d0
         yxi(i) = (Y(i+1)-Y(i-1))*0.5d0
      end do
         xxi(1) = X(2) -X(1)
         yxi(1) = Y(2) -Y(1)
         xxi(NI)= X(NI)-X(IM1)
         yxi(NI)= Y(NI)-Y(IM1)
      do i=1,NI
         xet(i) =-(yxi(i)*Fk(i))/((xxi(i))**2.0d0+(yxi(i))**2.0d0)
         yet(i) = (xxi(i)*Fk(i))/((xxi(i))**2.0d0+(yxi(i))**2.0d0)
      end do
c---------------------------------------- Area Blending
      area_mean = 0.0d0
      do i=1,NI
         area_mean = area_mean + Fk(i)
      end do
      area_mean = area_mean / NI
c------------------------------------------------------
c---- Matrix system DO I=1,IMAX
      do i=1,NI
c---- Calculate Cell Area
         if (i.eq.1) then
            dx1 = X(i+1)-X(i)
            dy1 = Y(i+1)-Y(i)
            delc= sqrt(dx1*dx1+dy1*dy1)
            a1  = X(i)-X(i+1)
            a2  = Y(i)-Y(i+1)
            b1  = X(i+1)-X(i)
            b2  = Y(i+1)-Y(i)
         else if (i.eq.NI) then
            dx1 = X(i)-X(i-1)
            dy1 = Y(i)-Y(i-1)
            delc= sqrt(dx1*dx1+dy1*dy1)
            a1  = X(i)-X(i-1)
            a2  = Y(i)-Y(i-1)
            b1  = X(i-1)-X(i)
            b2  = Y(i-1)-Y(i)
         else
            dx1 = X(i  )-X(i-1)
            dy1 = Y(i  )-Y(i-1)
            dx2 = X(i+1)-X(i  )
            dy2 = Y(i+1)-Y(i  )
            delc=0.5d0*(sqrt(dx1*dx1+dy1*dy1)+sqrt(dx2*dx2+dy2*dy2))
            a1  = X(i-1)-X(i)
            a2  = Y(i-1)-Y(i)
            b1  = X(i+1)-X(i)
            b2  = Y(i+1)-Y(i)
         end if
         alpha= dsqrt(a1*a1+a2*a2)*dsqrt(b1*b1+b2*b2)
         alpha= (a1*b1+a2*b2)/(alpha+1e-30)
         alpha= acos(alpha)
c        dels = dels + dels*dcos(alpha*0.5d0)
         F(i) = delc * dels
c-----------------------------------------Area Blending
         if(i.gt.2.and.i.lt.NI-1) then
         if (F(i).lt.area_mean*0.25d0) then
         factor = 0.5*(1.0d0+(1.0d0-factor_blend)**(j-2))
         F(i)   = factor*delc*dels + (1-factor)*area_mean*0.25d0
         end if
         end if
c        if(i.gt.101.and.i.lt.NI-100) then
c        if (F(i).gt.area_mean*3.0d0) then
c        factor = 0.5*(1.0d0+(1.0d0-factor_blend)**(j-2))
c        F(i)   = factor*delc*dels + (1-factor)*area_mean*3.0d0
c        end if
c        end if
c------------------------------------------------------
c        write(*,100) i,alpha,dels,F(i)
c---- Matrix system
         MatA(1,1) = xet(i)
         MatA(1,2) = yet(i)
         MatA(2,1) = yet(i)
         MatA(2,2) =-xet(i)

         MatB(1,1) =  xxi(i)
         MatB(1,2) =  yxi(i)
         MatB(2,1) = -yxi(i)
         MatB(2,2) =  xxi(i)

         MatI(1,1) = 1.0d0
         MatI(1,2) = 0.0d0
         MatI(2,1) = 0.0d0
         MatI(2,2) = 1.0d0

         MatH(1,1) = 0.0d0
         MatH(2,1) = F(i) + Fk(i)

         call Inv(MatB,InvB)
         call Mul(InvB,MatA,MatC,2,2)
         call Mul(InvB,MatH,MatD,2,1)

         AA(i,:,:) = -0.5d0*MatC(:,:)
         BB(i,:,:) = MatI(:,:)
         CC(i,:,:) =  0.5d0*MatC(:,:)
         DD(i,1,1) =  MatD(1,1) + X(i)
         DD(i,2,1) =  MatD(2,1) + Y(i)
      end do
c---------------------------------------- Boundary Condition
         dis = sqrt((X(2)-X(1))**2.0d0 +(Y(2)-Y(1))**2.0d0)
         dx  =-(Y(2)-Y(1))/dis*dels !0.5d0 ! delta_s
         dy  = (X(2)-X(1))/dis*dels !0.5d0 ! delta_s
         MatR(1,1) = X(1) + dx
         MatR(2,1) = Y(1) + dy
         RR(1,1,1) = MatR(1,1)
         RR(1,2,1) = MatR(2,1)
         MatA(:,:) = AA(2,:,:)
         call Mul(MatA,MatR,MatH,2,1)
         DD(2,1,1) = DD(2,1,1)-MatH(1,1)
         DD(2,2,1) = DD(2,2,1)-MatH(2,1)

         dis = sqrt((X(NI)-X(IM1))**2.0d0 +(Y(NI)-Y(IM1))**2.0d0)
         dx  =-(Y(NI)-Y(IM1))/dis*dels !0.5d0 ! delta_s
         dy  = (X(NI)-X(IM1))/dis*dels !0.5d0 ! delta_s
         MatR(1,1) = X(NI) + dx
         MatR(2,1) = Y(NI) + dy
         RR(NI,1,1) = MatR(1,1)
         RR(NI,2,1) = MatR(2,1)
         MatC(:,:) = CC(IM1,:,:)
         call Mul(MatC,MatR,MatH,2,1)
         DD(IM1,1,1) = DD(IM1,1,1)-MatH(1,1)
         DD(IM1,2,1) = DD(IM1,2,1)-MatH(2,1)
c----------------------------------------------------------- Smoothing
      do i=3,NI-2
         DD(i,1,1) = DD(i,1,1) +
     +           dc*(X(i+2)-4.d0*X(i+1)+6.d0*X(i)-4.d0*X(i-1)+X(i-2))
         DD(i,2,1) = DD(i,2,1) +
     +           dc*(Y(i+2)-4.d0*Y(i+1)+6.d0*Y(i)-4.d0*Y(i-1)+Y(i-2))
      end do

100   format(i4,4(f10.5,1x))
      return
      end
c------------------------------------------------------
