CC = nvcc
CFLAGS = -arch=sm_35 --ptxas-options -v
LFLAGS = -o
MAINFLAGS = -I -lboost_iostreams -lboost_system -lboost_filesystem


OBJ_FILES = $(wildcard ../Source/*.o)

QCD: $(OBJ_FILES)
	$(CC) $(CFLAGS) $(OBJ_FILES) $(LFLAGS) $@ $(MAINFLAGS)
