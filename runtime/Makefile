# sonar_edge C runtime. Dependency-free C11.
#   make            build libsonar_edge.a, the sonar-detect CLI and (with frontend.c)
#                   the end-to-end CLI sonar-edge-run
#   make test       bit-exact golden test against the Python integer simulation
#   make CC=aarch64-linux-gnu-gcc   cross-compile for a 64-bit Raspberry Pi
#   make CFLAGS="-std=c11 -O2 -Wall -Wextra -Werror -pedantic -mcpu=cortex-a7 -mfpu=neon-vfpv4"
#                   32-bit Raspberry Pi 2 with NEON (Raspbian's default target is ARMv6, no NEON)
#   make W=models/small_360_segdice   build another exported model (export_c --out DIR);
#                   run `make clean` when switching W
CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic
LDFLAGS ?=
W       ?=

ifneq ($(W),)
WEIGHTS := $(abspath $(W))/weights.h
GOLDEN  := $(abspath $(W))/tests/golden.h
WDEF    := -DSE_WEIGHTS_H='"$(WEIGHTS)"' -DSE_GOLDEN_H='"$(GOLDEN)"'
else
WEIGHTS := weights.h
GOLDEN  := tests/golden.h
WDEF    :=
endif

all: libsonar_edge.a sonar-detect $(if $(wildcard frontend.c),sonar-edge-run) $(if $(wildcard stream/sonar_stream.c),sonar-stream)

sonar_edge.o: sonar_edge.c sonar_edge.h $(WEIGHTS)
	$(CC) $(CFLAGS) $(WDEF) -c -o $@ $<

libsonar_edge.a: sonar_edge.o
	$(AR) rcs $@ $^

sonar-detect: cli/sonar_detect.c libsonar_edge.a
	$(CC) $(CFLAGS) -o $@ $< libsonar_edge.a $(LDFLAGS)

sonar-edge-run: cli/sonar_edge_run.c stream/stream_client.c stream/protocol.h frontend.h sonar_edge.h libsonar_edge.a
	$(CC) $(CFLAGS) -pthread -o $@ $< libsonar_edge.a $(LDFLAGS) -lm

tests/test_golden: tests/test_golden.c $(GOLDEN) libsonar_edge.a
	$(CC) $(CFLAGS) $(WDEF) -o $@ $< libsonar_edge.a

test: tests/test_golden
	./tests/test_golden

clean:
	rm -f *.o libsonar_edge.a sonar-detect sonar-edge-run sonar-stream tests/test_golden

.PHONY: all test clean

# Front end (M8): fan image or raw polar frame -> network input (frontend.h)
libsonar_edge.a: frontend.o

frontend.o: frontend.c frontend.h sonar_edge.h $(WEIGHTS)
	$(CC) $(CFLAGS) $(WDEF) -c -o $@ $<

tests/test_frontend: tests/test_frontend.c tests/frontend_golden.h libsonar_edge.a
	$(CC) $(CFLAGS) -o $@ $< libsonar_edge.a -lm

test-frontend: tests/test_frontend
	./tests/test_frontend

clean: clean-frontend
clean-frontend:
	rm -f tests/test_frontend

.PHONY: test-frontend clean-frontend

# ARIS recordings (.aris) -> polar frames for the front end (aris.h); beam tables from the SoundMetrics SDK
libsonar_edge.a: aris.o

aris.o: aris.c aris.h frontend.h $(wildcard beamwidths/*.h)
	$(CC) $(CFLAGS) -c -o $@ $<

# Live stream emulation (stream/protocol.h): sonar-stream replays recordings over TCP;
# sonar-edge-run --stream HOST:PORT is the real-time client.
sonar-stream: stream/sonar_stream.c stream/protocol.h aris.h libsonar_edge.a
	$(CC) $(CFLAGS) -o $@ $< libsonar_edge.a $(LDFLAGS) -lm
