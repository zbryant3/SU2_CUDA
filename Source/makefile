CC = nvcc
CFLAGS = -arch=sm_35 -std=c++11 -dc -o
LFLAGS =
MAINFLAGS =
SOURCE_FILES = $(wildcard *.cu)
OBJECTS = $(patsubst %.cu, %.o, $(SOURCE_FILES))
HEADER_FILES = $(wildcard Headers/*.cuh)

all: $(OBJECTS)
	@

%.o: %.cu $(HEADER_FILES)
	@$(CC) $(CFLAGS) $@ $< $(LFLAGS)



.PHONY: clean
clean:
	@rm $(OBJECTS)
