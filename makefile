SUBDIRS := Source Executables

OBJECTS = $(wildcard ./Source/*.o)
EXEC = ./Executables/AVGPLAQ

all:
	@$(MAKE) -w -C Source
	@$(MAKE) -w -C Executables

clean:
	@rm $(OBJECTS)
	@rm $(EXEC)
