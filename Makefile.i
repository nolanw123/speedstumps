# variables for generic make

SHELL=/bin/bash

CC=g++
CCSTD=--std=c++20

INCLUDES:=-I . $(INCLUDES)

CCFLAGS_generic+=-MD -fPIC $(CCSTD) -Wall $(INCLUDES)
CCFLAGS_debug+=-DDEBUG_BUILD -g $(CCFLAGS_generic)
CCFLAGS_opt+=-DRELEASE_BUILD -g -O3 $(CCFLAGS_generic)

LDFLAGS_generic+=
LDFLAGS_debug+=-g $(LDFLAGS_generic)  
LDFLAGS_opt+=$(LDFLAGS_generic)

BUILD_DIRS=debug opt

.PHONY: all checkdirs clean

all: buildall

buildall: checkdirs $(foreach lib,$(LIBS),$(foreach bdir,$(BUILD_DIRS),lib/$(bdir)/lib$(lib).a)) $(foreach bin,$(BINARIES),$(foreach bdir,$(BUILD_DIRS),exec/$(bdir)/$(bin))) $(foreach lib,$(DYLIBS),$(foreach bdir,$(BUILD_DIRS),lib/$(bdir)/lib$(lib).so))

checkdirs: $(foreach bdir,$(BUILD_DIRS),build_$(bdir))

clean:
	rm -rf objs exec lib

.SECONDEXPANSION:

define make-goal
objs/$1/%.o: %.cc
	$(CC) $(CCFLAGS_$(1)) -c $$< -o $$@
endef

define make-build-dir
build_$1:
	@mkdir -p lib/$(1)
	@mkdir -p objs/$(1)
	@mkdir -p exec/$(1)
endef

$(foreach bdir,$(BUILD_DIRS),$(eval $(call make-build-dir,$(bdir))))
$(foreach bdir,$(BUILD_DIRS),$(eval $(call make-goal,$(bdir))))

define make-bin
-include $$(patsubst %.cc,objs/$1/%.d,$($(2)_SRCS))
exec/$1/$2: $$(patsubst %.cc,objs/$1/%.o,$($(2)_SRCS)) $($2_BINDEPS)
	$(CC) $$^ $(LDFLAGS_$1) $($2_LIBS) $(SYS_LIBS) -o $$@
endef

$(foreach bin,$(BINARIES),$(foreach bdir,$(BUILD_DIRS),$(eval $(call make-bin,$(bdir),$(bin)))))

define make-lib
-include $$(patsubst %.cc,objs/$1/%.d,$($(2)_SRCS))
lib/$1/lib$2.a: $$(patsubst %.cc,objs/$1/%.o,$($(2)_SRCS))
	ar rcs $$@ $$^
endef

$(foreach lib,$(LIBS),$(foreach bdir,$(BUILD_DIRS),$(eval $(call make-lib,$(bdir),$(lib)))))

define make-dylib
-include $$(patsubst %.cc,objs/$1/%.d,$($(2)_SRCS))
lib/$1/lib$2.so: $$(patsubst %.cc,objs/$1/%.o,$($(2)_SRCS))
	$(CC) $$^ $(LDFLAGS_$1) $($2_LIBS) $(SYS_LIBS) -shared -o $$@
endef

$(foreach lib,$(DYLIBS),$(foreach bdir,$(BUILD_DIRS),$(eval $(call make-dylib,$(bdir),$(lib)))))

