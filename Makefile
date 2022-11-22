BINARIES=vectest vectest2

vectest_SRCS=vectest.cc

vectest2_SRCS=vectest2.cc

CCFLAGS_opt+=-mavx
CCFLAGS_debug+=-mavx

include Makefile.i

