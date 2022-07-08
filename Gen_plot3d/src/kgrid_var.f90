module kgrid_var

      integer*4 :: NI,NJ

      real(kind=8),allocatable,dimension(:,:,:) :: AA,BB,CC,DD,RR
      real(kind=8),allocatable,dimension(:)     :: X,Y,F,Fk
      real(kind=8),allocatable,dimension(:,:,:) :: Z
      real(kind=8) :: dis_first,dis_ratio,dc,cf

!     character(len=50) :: airname

!     -----------------------------------------------------------------
end module
