SUBDIRS := Source Executables

OBJECTS = $(wildcard ./Source/*.o)
EXEC = ./Executables/AVGPLAQ ./Executables/TIMER

all:
	@$(MAKE) -w -C Source
	@$(MAKE) -w -C Executables

clean:
	@rm $(OBJECTS)
	@rm $(EXEC)
