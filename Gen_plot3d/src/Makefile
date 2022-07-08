# makefile for kgrid
# For ALPHA LINUX
PROG = kgrid
SRCHD= kgrid_var.f90
SRC=   kgrid.f InitMat.f TDMA.f CalMat.f

OBJHD =  $(SRCHD:.f90=.o)
OBJ   =  $(SRC:.f=.o)

OBJS = $(OBJHD) $(OBJ)
LIBS =

CC = cc
CFLAGS =

#FC  = mpif90
FC  = /opt/mpi/intel/mpich-1.2.7p1/bin/mpif90
FFLAGS = -O2 -axSSSE3

#F90 = mpif90
F90 = /opt/mpi/intel/mpich-1.2.7p1/bin/mpif90
F90FLAGS = -O2 -axSSSE3

LDFLAGS = 

all: $(PROG) 

$(PROG): $(OBJS)
	$(F90) $(LDFLAGS) -o $@ $(OBJS) $(slv)
slv : 
	$(F90) $(FFLAGS) -c kgrid_var.f90 -o kgrid_var.o
clean:
	rm -f $(PROG) $(OBJS) *.mod
src:
	cat $(SRCS) > kgrid.src

.SUFFIXES: $(SUFFIXES) .f90 .f 

.f.o:
	$(F90) $(F90FLAGS) -c $<
.f90.o:
	$(F90) $(F90FLAGS) -c $<
ctags:
	ctags *.f *.f90

