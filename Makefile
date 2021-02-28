SHELL := /bin/bash # Use bash syntax
GCC = gcc
ARGS = -lm

CC = sm_20

NVCC = nvcc
NARGS = -lm

default: all

all: v0 v1 v2

v0: v0.c
	$(GCC) $(ARGS) -o $@ $^

v1: v1.cu
	$(NVCC) $(NARGS) -o $@ $^

v2: v2.cu
	$(NVCC) $(NARGS) -o $@ $^

clean:
	rm -rf *~ *.ptx v0 v1 v2
