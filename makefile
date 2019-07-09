SUBDIRS := Source Executables

OBJECTS = $(wildcard ./Source/*.o)
EXEC = ./Executables/Ising

all:
	$(MAKE) -w -C Source
	$(MAKE) -w -C Executables

clean:
	rm $(OBJECTS)
	rm $(EXEC)
