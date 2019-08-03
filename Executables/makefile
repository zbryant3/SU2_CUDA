CC = nvcc
CFLAGS = -arch=sm_35 --ptxas-options -v
LFLAGS = -o
MAINFLAGS =


AVGOBJ_FILES = ../Source/LattiCuda_Device.o ../Source/avgplaq.o ../Source/LattiCuda.o
TIMER_FILES = ../Source/LattiCuda_Device.o ../Source/timing.o ../Source/LattiCuda.o
TEST_FILES = ../Source/test.o ../Source/Complex.o

all: AVGPLAQ TIMER TEST

AVGPLAQ: $(AVGOBJ_FILES)
	@$(CC) $(CFLAGS) $(AVGOBJ_FILES) $(LFLAGS) $@ $(MAINFLAGS)

TIMER: $(TIMER_FILES)
		@$(CC) $(CFLAGS) $(TIMER_FILES) $(LFLAGS) $@ $(MAINFLAGS)

TEST: $(TEST_FILES)
	@$(CC) $(CFLAGS) $(TEST_FILES) $(LFLAGS) $@ $(MAINFLAGS)
