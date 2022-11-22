BINARIES=vectest vectest2

vectest_SRCS=vectest.cc

vectest2_SRCS=vectest2.cc

CCFLAGS_opt+=-mavx -mavx2
CCFLAGS_debug+=-mavx -mavx2

include Makefile.i

