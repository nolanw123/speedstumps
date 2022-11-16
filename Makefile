BINARIES=vectest

vectest_SRCS=vectest.cc

CCFLAGS_opt+=-mavx
CCFLAGS_debug+=-mavx

include Makefile.i

