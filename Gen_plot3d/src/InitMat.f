c------------------------------------------------------
      subroutine InitMat
      use kgrid_var
      implicit none
      integer*4 :: i,icase
      integer*4 :: m,mi,mj,mk
      integer*4 :: mi1,mj1,mk1
      integer*4 :: mi2,mj2,mk2
      integer*4 :: mi3,mj3,mk3
      integer*4 :: mi4,mj4,mk4

      icase = 3
      if (icase.eq.1) then
         open(5,file='Case1.x')
         read(5,*) m
         read(5,*) mi,mj,mk
         mj = 40
         NI = mi ; NJ = mj
         dis_first = 0.02d0
         dis_ratio = 0.01d0
         dc        = 0.00d0 !0.02d0
         allocate( Z(NI,NJ,2))
         allocate(AA(NI, 2,2))
         allocate(BB(NI, 2,2))
         allocate(CC(NI, 2,2))
         allocate(DD(NI, 2,1))
         allocate(RR(NI, 2,1))
         allocate( X(NI))
         allocate( Y(NI))
         allocate( F(NI))
         allocate(Fk(NI))
         AA=0.0d0 ; BB=0.0d0 ; CC=0.0d0 ; DD=0.0d0 ; RR=0.0d0
         Z =0.0d0 ; X =0.0d0 ; Y =0.0d0 ; F =0.0d0 ; Fk=0.0d0
         read(5,*) (Z(i,1,1),i=1,NI)
         read(5,*) (Z(i,1,2),i=1,NI)
         close(5)
      else if (icase.eq.2) then
         open(5,file='Case2.x')
         read(5,*) m
         read(5,*) mi,mj,mk
         mj = 30
         NI = mi ; NJ = mj
         dis_first = 0.007d0
         dis_ratio = 0.01d0
         dc        = 0.00d0
         allocate( Z(NI,NJ,2))
         allocate(AA(NI, 2,2))
         allocate(BB(NI, 2,2))
         allocate(CC(NI, 2,2))
         allocate(DD(NI, 2,1))
         allocate(RR(NI, 2,1))
         allocate( X(NI))
         allocate( Y(NI))
         allocate( F(NI))
         allocate(Fk(NI))
         AA=0.0d0 ; BB=0.0d0 ; CC=0.0d0 ; DD=0.0d0 ; RR=0.0d0
         Z =0.0d0 ; X =0.0d0 ; Y =0.0d0 ; F =0.0d0 ; Fk=0.0d0
         read(5,*) (Z(i,1,1),i=1,NI)
         read(5,*) (Z(i,1,2),i=1,NI)
         close(5)
      else if (icase.eq.3) then
         write(*,*) 'Test1'
         open(5,file='Case3.x')
         read(5,*) m
         read(5,*) mi1,mj1,mk1
         read(5,*) mi2,mj2,mk2
         read(5,*) mi3,mj3,mk3
         read(5,*) mi4,mj4,mk4
         mj = 81
         NI = mi1+mi2+mi3+mi4-3 ; NJ = mj
         dis_first = 0.0001d0
         dis_ratio = 0.116d0
         dc        = 0.00d0
         allocate( Z(NI,NJ,2))
         allocate(AA(NI, 2,2))
         allocate(BB(NI, 2,2))
         allocate(CC(NI, 2,2))
         allocate(DD(NI, 2,1))
         allocate(RR(NI, 2,1))
         allocate( X(NI))
         allocate( Y(NI))
         allocate( F(NI))
         allocate(Fk(NI))
         AA=0.0d0 ; BB=0.0d0 ; CC=0.0d0 ; DD=0.0d0 ; RR=0.0d0
         Z =0.0d0 ; X =0.0d0 ; Y =0.0d0 ; F =0.0d0 ; Fk=0.0d0
         read(5,*) (Z(i,1,1),i=1,mi1)
         read(5,*) (Z(i,1,2),i=1,mi1)
         read(5,*) (F(i)    ,i=1,mi1)
         read(5,*) (Z(i,1,1),i=mi1,mi1+mi2-1)
         read(5,*) (Z(i,1,2),i=mi1,mi1+mi2-1)
         read(5,*) (F(i)    ,i=mi1,mi1+mi2-1)
         read(5,*) (Z(i,1,1),i=mi1+mi2-1,mi1+mi2+mi3-2)
         read(5,*) (Z(i,1,2),i=mi1+mi2-1,mi1+mi2+mi3-2)
         read(5,*) (F(i)    ,i=mi1+mi2-1,mi1+mi2+mi3-2)
         read(5,*) (Z(i,1,1),i=mi1+mi2+mi3-2,mi1+mi2+mi3+mi4-3)
         read(5,*) (Z(i,1,2),i=mi1+mi2+mi3-2,mi1+mi2+mi3+mi4-3)
         read(5,*) (F(i)    ,i=mi1+mi2+mi3-2,mi1+mi2+mi3+mi4-3)
         close(5)

         write(*,*) 'Test2'


      end if

      return
      end
c------------------------------------------------------
