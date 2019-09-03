SUBDIRS := Source Executables

OBJECTS = $(wildcard ./Source/*.o)
EXEC = ./Executables/AVGPLAQ ./Executables/TIMER ./Executables/POLYKOV ./Executables/TEST

all:
	@$(MAKE) -s -w -C Source
	@$(MAKE) -s -w -C Executables

clean:
	@rm $(OBJECTS)
	@rm $(EXEC)
