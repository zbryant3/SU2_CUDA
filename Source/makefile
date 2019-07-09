CC = nvcc
CFLAGS = -arch=sm_35 -dc -o
LFLAGS =
MAINFLAGS = -I -lboost_iostreams -lboost_system -lboost_filesystem

SOURCE_FILES = $(wildcard *.cu)
OBJECTS = $(patsubst %.cu, %.o, $(SOURCE_FILES))
HEADER_FILES = $(wildcard Headers/*.cuh)

all: $(OBJECTS)

main.o: main.cu $(HEADER_FILES)
	$(CC) $(CFLAGS) $@ $< $(LFLAGS) $(MAINFLAGS)

%.o: %.cu $(HEADER_FILES)
	$(CC) $(CFLAGS) $@ $< $(LFLAGS)



.PHONY: clean
clean:
	rm $(OBJECTS)
