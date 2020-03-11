SHELL = /bin/sh
CC=mpicc
CFLAGS=-Wall -Ofast -DMKL_ILP64 -march=native
CLIBS=-lmkl_scalapack_ilp64 -lmkl_blacs_openmpi_ilp64 \
	-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 \
	-lpthread -lm
CMAKE_PATH=-L/opt/intel/compilers_and_libraries_2019/linux/mkl/lib/intel64_lin \
		-L/opt/intel/compilers_and_libraries_2019/linux/compiler/lib/intel64_lin
CMAKE_INCLUDE=-I/opt/intel/compilers_and_libraries_2019/linux/mkl/include
DEPS=debug_util.h randsvd_large.h
OBJ=debug_util.o randsvd_large.o

all: test_orthogonal test_conditioning test_timing

test_%: test_%.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(CLIBS)

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm test_orthogonal test_conditioning  test_timing
	rm *.e* *.o* *~
